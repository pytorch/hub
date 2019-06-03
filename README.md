# Pytorch Hub

[![CircleCI](https://circleci.com/gh/pytorch/hub.svg?style=svg)](https://circleci.com/gh/pytorch/hub)

## Logistics

We accept submission to Pytorch hub through PR in `pytorch/hub` repo. Once the PR is merged into master here, it will show up on Pytorch website in 24 hrs.

## Steps to submit to Pytorch hub

1. Add a `hubconf.py` in your repo, following the instruction in [torch.hub doc](https://pytorch.org/docs/master/hub.html#publishing-models). Verify it's working correctly by running `torch.hub.load(...)` locally.
2. Create a PR in `pytorch/hub` repo. For each new model you have, create a `<repo_owner>_<repo_name>_<title>.md` file using this [template](docs/pull_request_template.md).

### Notes
- Currently we don't support hosting pretrained weights, users with pretrained weights need to host them properly themselves.
- In general we recommend one model per markdown file, models with similar structures like `resnet18, resnet50` should be placed in the same file.
- If you have images, place them in `images/` folder and link them correctly in the `[images/featured_image_1/featured_image_2]` fields above.
- We only support a pre-defined set of tags, currently they are `{nlp, vision, audio, generative}`. We will expand this set as needed.
- To test your PR locally, run `python scripts/sanity_check.py` and `./scripts/run_pytorch.sh`.
- Our CI concatenates all code blocks in one markdown file and runs it agaist the latest Pytorch-cpu release. If your `dependencies` is not installed on our CI machine, add them in [install.sh](scripts/install.sh).
- We provide a way to preview your model webpage through `netlify bot`. This bot builds your PR with latest `pytorch.github.io` repo and comment on your PR with preview link. The preview will be updated as you push more commits to the PR.
![Example netlify bot comment](images/netlify.png)

