"""
setup_midas.py

Run this once after `pip install -r requirements.txt`:

    python setup_midas.py

What it does
------------
Vendors (locally clones) the MiDaS repository and its nested
EfficientNet backbone dependency, then pre-trusts both with
torch.hub, so that `measure_server.py` can load the depth model via
`torch.hub.load(..., source="local")` instead of going through
torch.hub's normal GitHub-API-backed download path.

Why this is necessary (not just a convenience)
------------------------------------------------
`torch.hub.load("intel-isl/MiDaS", ...)` does three network-dependent
things before it ever gets to actually downloading model weights:

  1. Calls the GitHub REST API to check whether the requested ref is
     a branch or a tag (`_parse_repo_info`).
  2. Calls the GitHub REST API again to confirm the repo isn't a
     malicious fork (`_validate_not_a_forked_repo`) - this runs even
     when `trust_repo=True`, because `trust_repo` only controls
     whether you're *prompted*, not whether this check runs at all.
  3. MiDaS's own hubconf.py internally calls torch.hub.load() a
     *second* time, on a completely different repo
     (rwightman/gen-efficientnet-pytorch) to fetch its backbone. That
     nested call doesn't inherit your trust_repo argument, so on a
     fresh cache it either hits an interactive y/N prompt (which
     crashes a non-interactive server with EOFError) or repeats steps
     1-2 against the second repo.

On networks that block or rate-limit the GitHub REST API specifically
(common on corporate networks, some CI runners, and sandboxed
environments) - while still allowing plain HTTPS git operations - step
2 fails with an HTTP error. A bug in torch's own error handling
(it unconditionally tries to delete a response header that's only
present on some failure paths) then turns that into a confusing
`KeyError: 'Authorization'` instead of a clear network error, which is
what motivated digging into this rather than assuming MiDaS itself was
broken.

This script sidesteps all of it: a plain `git clone` only needs HTTPS
access to github.com itself, not its REST API, so it succeeds in
exactly the environments where torch.hub's own download path fails.
"""

import os
import shutil
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(HERE, "vendor")
MIDAS_DIR = os.path.join(VENDOR_DIR, "MiDaS")

# torch.hub caches nested repo dependencies under
# <hub_dir>/<owner>_<repo>_<branch>. gen-efficientnet-pytorch's
# default branch is "master" (no "main" branch exists in that repo).
NESTED_REPO_URL = "https://github.com/rwightman/gen-efficientnet-pytorch.git"
NESTED_REPO_CACHE_NAME = "rwightman_gen-efficientnet-pytorch_master"


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def clone_midas():
    hubconf_path = os.path.join(MIDAS_DIR, "hubconf.py")
    if os.path.isfile(hubconf_path):
        print(f"MiDaS already vendored at {MIDAS_DIR}, skipping clone.")
        return
    if os.path.isdir(MIDAS_DIR):
        # A directory exists but hubconf.py doesn't - this is a
        # leftover from a previous failed/interrupted clone (e.g. a
        # network drop partway through). Checking isdir() alone here
        # is what originally caused this: the script would see the
        # directory, assume the clone had succeeded, and skip cloning
        # forever afterward - leaving a permanently broken vendor/
        # folder that no amount of re-running setup_midas.py would
        # fix. Remove the incomplete clone and retry instead of
        # trusting that "the folder exists" means "the clone worked."
        print(f"Found an incomplete clone at {MIDAS_DIR} (missing hubconf.py) - removing and retrying.")
        shutil.rmtree(MIDAS_DIR)
    os.makedirs(VENDOR_DIR, exist_ok=True)
    run(["git", "clone", "--depth", "1", "https://github.com/isl-org/MiDaS.git", MIDAS_DIR])
    if not os.path.isfile(hubconf_path):
        raise RuntimeError(
            f"git clone reported success but {hubconf_path} is still missing. "
            "The clone may have completed against a corrupted or unexpected repo state."
        )


def vendor_nested_efficientnet_repo():
    """
    Pre-populate torch.hub's cache directory with a direct clone of
    the nested EfficientNet backbone repo, at the exact path
    torch.hub expects, so it's treated as already-cached and torch.hub
    never has to reach the GitHub API for it at all.
    """
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)
    target = os.path.join(hub_dir, NESTED_REPO_CACHE_NAME)
    hubconf_path = os.path.join(target, "hubconf.py")

    if os.path.isfile(hubconf_path):
        print(f"Nested EfficientNet repo already cached at {target}, skipping.")
        return
    if os.path.isdir(target):
        # Same incomplete-clone scenario as clone_midas() above - don't
        # trust an existing directory without verifying it actually
        # contains what we expect.
        print(f"Found an incomplete cache at {target} (missing hubconf.py) - removing and retrying.")
        shutil.rmtree(target)

    tmp_clone = target + ".tmp-clone"
    if os.path.isdir(tmp_clone):
        shutil.rmtree(tmp_clone)

    run(["git", "clone", "--depth", "1", NESTED_REPO_URL, tmp_clone])
    # torch.hub's own cached clones don't carry a .git directory; match
    # that so nothing downstream treats this as an editable repo.
    shutil.rmtree(os.path.join(tmp_clone, ".git"), ignore_errors=True)
    os.rename(tmp_clone, target)
    if not os.path.isfile(hubconf_path):
        raise RuntimeError(
            f"git clone reported success but {hubconf_path} is still missing."
        )
    print(f"Vendored nested EfficientNet repo to {target}")


def pre_trust_repos():
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)
    trusted_list_path = os.path.join(hub_dir, "trusted_list")
    existing = set()
    if os.path.exists(trusted_list_path):
        with open(trusted_list_path) as f:
            existing = {line.strip() for line in f if line.strip()}
    to_add = {"intel-isl_MiDaS", "rwightman_gen-efficientnet-pytorch"} - existing
    if to_add:
        with open(trusted_list_path, "a") as f:
            for entry in to_add:
                f.write(entry + "\n")
        print(f"Pre-trusted: {sorted(to_add)}")
    else:
        print("Repos already in torch.hub's trusted_list.")


def verify_load():
    print("\nVerifying the model actually loads end-to-end (this downloads weights, ~50MB)...")
    try:
        model = torch.hub.load(MIDAS_DIR, "MiDaS_small", source="local", trust_repo=True)
        print("Success:", type(model).__name__)
        return True
    except Exception as e:
        print(f"\nModel weights download failed: {e!r}")
        print(
            "This is the actual model *weights* download (hosted on GitHub "
            "Releases, which redirects to a separate CDN domain at request "
            "time) failing - a plain network/firewall issue on whatever "
            "machine ran this script, not the torch.hub API issue this "
            "script exists to work around (that part - the GitHub API calls "
            "and the nested-repo trust prompt - already succeeded above, or "
            "this script would have failed earlier with a different error).\n"
            "\n"
            "To resolve it:\n"
            "  - Run this script on a network that can reach "
            "*.githubusercontent.com (not just github.com itself), or\n"
            "  - Download tf_efficientnet_lite3-b733e338.pth manually from\n"
            "    https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-weights\n"
            "    and place it at ~/.cache/torch/hub/checkpoints/\n"
            "\n"
            "The vendored MiDaS/EfficientNet *code* is already in place either "
            "way - this only affects whether the pretrained weights can be "
            "fetched. The app still runs fine without depth correction."
        )
        return False


if __name__ == "__main__":
    try:
        clone_midas()
        pre_trust_repos()
        vendor_nested_efficientnet_repo()
        if verify_load():
            print("\nDone. measure_server.py will now load MiDaS from the local vendor/ clone.")
        else:
            print(
                "\nSetup partially complete: torch.hub's API/trust issues are worked "
                "around, but weights still need to be fetched (see above). "
                "measure_server.py will retry the weights download on its own next "
                "time it starts.",
            )
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\ngit command failed: {e}", file=sys.stderr)
        print(
            "This script needs `git` installed and HTTPS access to github.com. "
            "If your network blocks that too, MiDaS depth correction won't be "
            "available, but the app still works without it (width-only "
            "estimation, slightly lower confidence score).",
            file=sys.stderr,
        )
        sys.exit(1)
