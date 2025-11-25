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

To get things to compile on Windows, it looks like using python 3.12 might be a good bet. Tbd whether this actually works though. It doesn't...

Let's get pyvista integrated, that'll be nice. Then after that the only difficulty may be in generating images. But I don't think that'll be too bad.

Having a look at the Boe paper, I think I can do a lot better with measuring further out as appropriate. Some of them are really rather nice! And it's probably wise to use a proven collection by someone else.

Tuesday 11th November:

Plan for today:

- Add something which saves out the raw crot downloads so I stop getting blocked from the server.
- Add the image generation to the tracer, and some optional flags like not plotting the occasional open field line.

I think that's probably sufficient.
The first of these goals has been acheived. Good good.

Notes for the image generator:
- Will be a square view but would like to make the dimensions variables, along with the resolution. The default values for these can be specified in the python wrapper rather than trying to be snazzy with the Fortran.
- Get the Fortran to output two arrays -- both the field lines and the image. But either of them have the potential to be none, which needs to be caught in the Python.


Wednesday 12th November:

The aims from yesterday have been roughly acheived. Probably not worth writing tests for the image generation, but maybe at some point... Alas still can't run linux stuff locally and my IP is still blocked by JSOC. Not sure how long I'll have to wait for that one.

Thursday 13th November:

The images have converged resonably nicely with 10,000 field lines. I'm going to experiment with getting openmp to work so it can trace field lines more effectively, but perhaps mext week would be more wise. Also would need to automate checking whether openmp is even an option.

The parameters from the 10,000 run are 
[-0.203,0.587,-1.063,0.002,-0.497,-0.831]

I've got it running with Openmp -- now will do the same with 50,000 lines using those parameters as a base

python -m numpy.f2py -c fortran/outflow_calc.f90 -m outflow_calc

ON Hamilton:
keep venv on nobackup to do image things

module load python
python -m venv /nobackup/vgjn10/projects/outflowpy/.venv
source /nobackup/vgjn10/projects/outflowpy/.venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install .


module load python/3.13.9

Monday 17th November:

Run with 50,000 lines has converged nicely. Making a video from it now...

Resulting parameter set:

[-0.135,0.764,-0.667,-0.003,-1.38,-1.359]

Nothing too extreme, which is good.

With 250,000 lines, the parameters are quite different:

[-0.003,0.533,-0.165,0.009,-2.314,-2.625]

but the result looks pretty good. That might be the most interesting of the videos, and those are the parameters we should probably use. Interesting that the parameters seem to follow a nice pattern here!

Perhaps it's worth doing some manual image manipulation on this as well to get them to match even better. Could definitely be quite snazzy about it.

Wednesday 19th:

Doing it manually wasn't going to work, so I automated it instead, which seems to have worked pretty well!

Optimums with 50,000:

[-0.023,0.42,-0.746,1.084]

and with 250,000:

[-0.033,0.635,0.285,-5.356],

which is pretty extreme skewing! Cool. Probably would be even more so for more field lines? Notable that the magnetic field strength at a point is the only thing that is really zero.

It's probably worth compiling the whole set of images and getting some average values to use for the 'image match' algorithm. Perhaps the distribution could be hard-coded but that sounds tricksy and a bit ugly. TBD.

NASA have unblocked me. Lovely. That means can run the every-day plots again, providing I make sure that the caching is working...

Maybe should add extra smoothing to images after the scaling? Getting some strange artefacts sometimes otherwise

Thursday 20th November:

I've now fixed the image matching and some of them are looking pretty good! I think it's probably worth doing the optimisation runs again (perhaps on ehrenfest) with a random sample from all the eclipses and 100,000 lines. It'll definitely take longer but hopefully not too problematic.

Doing some pyvista plots to check all the solutions are actually reasonable. 2006 in particular looks very odd.

Friday 21st November:

Have done some runs looking at all eclipses randomly (on ehrenfest) and just the 2012 one on airy. Looking OK but a way off what it should look like. The 2008 one is still pretty great.

Parameters from 2012 run:

[-0.157,-0.002,0.23,1.844],

which have converged pretty nicely.

I think perhaps some manual intervention may be required. When it's not a dipole the image similarity thing can't tell what's what. 

Until midday (maximum), let's investigate why the dipole ones worked quite well and see what's wrong with the others?

Monday 24th November:

I've had an idea to do log scaling and THEN the matching, to avoid the oversaturation problem. Should also look into making the resolutions more flexible, so can do some nice high res ones which look all nice like.

That's made very little difference, but does avoid any arbitrary cutoffs at the top - should look nicer!

I think perhaps an image optimisation run on one of the messy eclipses (2012) on ehrenfest might be wise, and hopefully wouldn't take too long. 

Also should determine the optimum number of field lines. 1000000 is clearly too many, but 10000 too few. 50,000 seems OK but it's probably better to test scientifically with the comparison code. Only one dimension so not too bad. By eye 50,000 seems fine so let's stick with that.

nohup python 3_image_optimisation.py >& output.log &

Have set it running on ehrenfest for the 2019 eclipse. Hopefully will look nice...
 
Tuesday 25th November:

Does indeed look quite nice. The 'best' parameters are:

[0.216,0.377,-0.323,1.567]. It's not quite converged yet but not far off.

I'll make some eclipse pictures based on that. Perhaps it might be wise to save the picture distribution for 2017 or something so one doesn't need an image reference? That won't take long. 
