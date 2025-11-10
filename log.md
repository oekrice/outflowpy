Starting new log for progress with the outflowpy package.

Tuesday 4th November:

pytest is now running on the pfsspy tests, apart from those which involve downloading data. I'll need to figure out a way of recreating these for my HMI/MDI process. Also might be necessary/nice to get it working for gong too, to be fair. 

Today plan: 

- Write some tests for the download of a single CR, both MDI and HMI
- Integrate my code to figure out the magnetic field at a precise date
- Test the smoothing code? Not sure how, but we'll see.

To run pytest need to activate the virtual environment twice, for some reason.

Wednesday 5th November:

- Integrate the code for precise timing (necessary for a long run), with unit tests in download_tests as per (woop)
- Find the runbatch code which can be used for the first bit of a paper. Look into correlations with the two parameters
- Install and run on Hamilton?

python -m venv ./.venv/

Thursday 6th November:

Github authentication with tokens:

ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub

Copy and paste into github then

git remote set-url origin git@github.com:<username>/<repo>.git
ssh -T git@github.com

- Remote plotting is now working and the package is nicely running on Hamilton
- Let's figure out the correlation plots next. Not sure where I put them.
- Also look into f2py-ing the tracing code, looking to compile it into the package at some point.

Monday 10th November:

I'm not sure what happened on Friday, but I think it was mainly getting the tracer integrated and working nicely. Alas I got banned from downloading data, but that appears to have subsided now. I'll try again but a little less forcefully...


