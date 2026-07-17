# I Taught an AI to Recognize the Shadows of Four-Dimensional Objects

*How a blurry Pentagon UFO video turned into a dataset, a detector, and an
arXiv paper, built entirely on a laptop with a 4 GB GPU.*

---

## It started with a weird video

A few years ago the Pentagon released a batch of UAP footage. One clip
showed a strange star-shaped object drifting across the night sky, and the
internet did what the internet does: aliens, obviously.

My first thought was different. Let's assume, just for a moment, that it
really was something alien. Would we be seeing the actual thing, or only
the version we humans are equipped to perceive, in 3D? That led to a
sharper question: how could we ever know whether something we see is a
true 3D object, or the shadow of something higher-dimensional?

Hold a cube in front of a lamp. Its shadow on the wall might be a square, a
hexagon, or something in between, depending on how you turn it. The shadow
is genuinely two-dimensional, every point of it lies flat on the wall, but
it is not an honest 2D object. It is a 3D object wearing a 2D disguise.

So I wondered: if something four-dimensional ever moved through our world,
we would not see it as it is. We would see its 3D shadow. And that shadow
would look, for lack of a better word, weird.

(For the record: that star-shaped object was later convincingly explained
as bokeh, an out-of-focus light source filtered through a triangular camera
aperture. The boring explanation won, as it almost always does. Hold onto
that thought, it comes back at the end.)

The question that stayed with me was not "was that a UFO." It was sharper
and more interesting:

**If a higher-dimensional object cast a shadow into our world, could we
even tell? Is a 4D shadow measurably different from an ordinary 3D thing?**

I searched the literature. Vision scientists had studied whether humans can
perceive 4D structure. Mathematicians had written about projections for a
century. But nobody had ever built a dataset, trained a detector, or put a
number on it. The question was sitting there, untouched.

So I did it myself.

## Wait, how do you even make a 4D object?

Here is the trick that makes the whole project possible: computers do not
care how many dimensions your data has.

A point in 3D space is three numbers: (x, y, z). A point in 4D space is
just four numbers: (x, y, z, w). NumPy will happily store a million of
them. You cannot *look* at them directly, but you can do math with them,
rotate them, and most importantly, project them.

Projection is what a lamp does to a cube. Mathematically, the simplest
version is almost embarrassing: to project a 4D point down to 3D, you
delete the fourth number. That is it. The shadow of a 4D object is what
remains when one coordinate is thrown away, exactly the way a cube's
shadow is what remains when depth is thrown away.

So the recipe for the dataset, which I called **HyperShadow**, is:

1. **Generate ordinary 3D shapes**: spheres, solid balls, donuts, cubes,
   cylinders, ellipsoids, even a tube wrapped around a trefoil knot.
   These get the label "native 3D."
2. **Generate genuinely higher-dimensional shapes**: the 4D sphere, the
   tesseract (the 4D cube you have seen spinning in YouTube animations),
   the Clifford torus, 5D and 6D objects, and randomly generated smooth
   4D-6D surfaces so the collection never repeats itself.
3. **Rotate each higher-dimensional object randomly in its own space**,
   then project it down to 3D. These get the label "shadow."
4. Ask: can anything tell the two classes apart?

The final dataset is 10,800 point clouds of 1,024 points each, plus 1,800
short "movies" of rotating objects. All of it generates from two random
seeds on a normal CPU in minutes. No supercomputer anywhere in this story.

## The paranoid part (which is the most important part)

Any machine learning person reading this is already suspicious, and they
should be. Synthetic datasets are notorious for hidden shortcuts: the model
"solves" the task by noticing something dumb, like one class being slightly
bigger, and you fool yourself into thinking it learned something deep.

I spent more time closing loopholes than building models. The rules:

- **Same treatment for both classes.** The ordinary 3D shapes get random
  rotations too, and the same noise and corruption as the shadows. Nothing
  about the preprocessing distinguishes the classes.
- **No free information.** Every cloud is centred at the origin, rescaled
  to the same average radius, and resampled to exactly 1,024 points. Size,
  position, and point count carry zero signal.
- **The killer rule: volume is not the answer.** Here is a fact that
  surprised me when I first plotted it. The shadow of a 4D sphere is not a
  hollow sphere. It is a **solid ball**. Every interior point gets filled
  in. So if the dataset only had hollow 3D shapes, a model could win by
  answering "is it filled?" To block this, I put a genuine solid ball in
  the ordinary-3D class. Now "filled" is not enough. The only difference
  between a real ball and a hypersphere's shadow is *where the mass
  concentrates inside*, and it takes real statistical work to see it.
- **Difficulty tiers.** Each cloud comes in four versions: clean, noisy,
  partially occluded, and corrupted with sensor-style dropout, so methods
  can be stress-tested the way real data would stress them.

## Experiment one: the obvious tool fails, and the failure is beautiful

If you ask a data scientist "how do I check the dimension of my data," they
will point you at intrinsic-dimension estimators, classic algorithms like
TwoNN and maximum-likelihood estimation. These look at how points crowd
around their neighbours and infer the dimensionality of whatever surface
the data lives on.

I ran them on HyperShadow. They scored 72 to 74 percent. Barely better
than guessing.

And once you see why, you cannot unsee it: **a shadow is still
three-dimensional data.** Every point of the shadow lives in 3D. There is
no fourth coordinate left to detect; projection destroyed it. Asking "what
dimension is this data" gives an honest answer, "at most three," which is
true and useless.

The dimension is not the clue. The clue is what projection *does* to the
object on the way down: it folds density into telltale patterns, fills
volumes with a specific radial profile, and scrambles topology. The
evidence of the fourth dimension is not in the dimension. It is in the
fingerprints of the squashing.

This, to me, is the most quotable finding of the paper: *dimensionality is
the wrong observable.*

## Experiment two: a small neural network sees the fingerprints

Next I trained a deliberately tiny PointNet-style network. 190,914
parameters, which by 2026 standards is microscopic, and about four minutes
of training on my GTX 1650 Ti.

Result: **96.2 percent accuracy** (plus or minus 0.3 over five training
runs), degrading gracefully from 98 percent on clean data to 94 percent on
the heavily corrupted tier.

Two details matter more than the headline number.

First, **it generalizes to shapes it has never seen.** I retrained the
model with entire object families deleted from the training set, then
tested only on the deleted family. Hypertori it had never encountered:
detected at 91 percent. Hypercubes: 79 percent. The model did not memorize
my shape catalogue. It learned what *projection itself* looks like.

Second, **its mistakes are exactly the ones mathematics predicts.** Its
worst confusion, by far, is the solid ball versus the 4D sphere's shadow,
the pair that differs only in interior density. When a model's failure
modes line up with theory, you can start to trust that the benchmark is
measuring something real.

## Experiment three: motion betrays the fourth dimension (my favourite result)

Everything above uses still snapshots. The best detector in the paper uses
none of that machinery. It uses motion, and it has **zero trained
parameters**.

Here is the idea. Take a rigid 3D object, a rock, a chair, anything, and
rotate it. Every frame of that motion can be perfectly explained by a
rigid 3D transformation: some rotation plus some shift. That is
essentially the definition of "rigid."

Now take a rigid 4D object, rotate it rigidly in 4D, and watch its 3D
shadow. The shadow *morphs*. Parts swell, shrink, and flow through each
other, like the spinning tesseract animations where cubes turn inside out.
The object is perfectly rigid. Its shadow is not, and, this is the crucial
part, **no rigid 3D motion can explain the shadow's frame-to-frame
changes.**

That impossibility is measurable. There is a closed-form formula from the
1970s, the Kabsch algorithm, that computes the best possible rigid
alignment between two point sets and tells you the leftover error. For a
real 3D object in rigid motion, the leftover is basically zero. For the
shadow of a rotating 4D object, it cannot be zero.

I computed that one number per sequence. Real 3D motion: residual about
0.02. Shadows: about 0.10, four times higher. A single threshold on this
single number separates the classes with **AUROC 0.982**. No neural
network. No training. A fifty-year-old formula and a ruler.

I call this a **rigidity witness**, and it is the idea I care most about,
because of its logical shape. A large residual does not merely *suggest*
higher-dimensional origin. It *rules out* every rigid-3D explanation at
once. You never observe the fourth dimension. You certify that the
three-dimensional story cannot be true. Physics fans will recognize the
family resemblance to Bell inequalities, which never photographed hidden
variables, they made an entire class of worldviews untenable with one
measured number.

Also worth savouring: motion reveals what snapshots hide. A 2023 vision
science study found that humans can distinguish rigid from non-rigid
motion of a hypercube's projection too. Whatever our visual system is
doing, it seems to have stumbled onto a version of the same trick.

## So... did I find 4D aliens?

No. And the paper says so in bold letters, because this is where projects
like this one live or die.

Everything above was tested on simulated geometry, where I knew the ground
truth. That is the only honest way to start: you validate the instrument
where the answer is known. What the work establishes is precisely this:

1. Shadows of higher-dimensional objects carry detectable statistical
   fingerprints, even under noise and occlusion.
2. The standard dimensionality tools cannot see those fingerprints.
   Learned models and motion-based tests can.
3. There exist simple, interpretable numbers, dimensional witnesses, that
   are provably near zero under any rigid 3D explanation, so a large value
   rules that entire class of explanations out.

What it does not establish is anything about physical reality. If someone
someday points tools like this at real tracked data and the witness fires,
the correct conclusion is "no rigid 3D model of the tested kind explains
this," followed by a long, boring, necessary hunt through every mundane
cause: tracking errors, non-rigid objects, optics, atmospherics.

Remember the bokeh. The star-shaped UFO that started this whole project
turned out to be a camera artifact, and the person who figured that out
did better science that day than anyone shouting about aliens. If this
project has a moral, it is that you can take a fringe-sounding question
seriously *by being stricter than everyone else about it*, not looser.

## What surprised me about doing this solo

- **The hardware was never the bottleneck.** One laptop, 4 GB of GPU
  memory, 8 GB of RAM (and yes, it crashed from memory pressure more than
  once). Full training run: four minutes. Every number in the paper
  reproduces overnight on hardware most gamers would call outdated.
- **The design work mattered more than the model.** The fairness rules
  took longer than the neural network. They are also the reason the
  results survive scrutiny.
- **Finding a rival paper was the best thing that happened.** Midway
  through, I found the 2023 human-perception study of 4D rigidity. For an
  hour it felt like being scooped. Then I realized: no dataset, no
  algorithm, no benchmark existed. Their work proved the question was
  respectable, and mine gives their humans a machine to compete with.
- **There are more questions like this.** Too strange for a grant, too
  small for a lab, exactly right for one stubborn person with a consumer
  GPU. I suspect there are hundreds of them lying around.

## Try it yourself

Everything is public, seeded, and reproducible:

- **Paper (arXiv):** https://arxiv.org/abs/2607.14419
- **Code and tests:** https://github.com/AkshaySasi/hypershadow
- **Dataset (Hugging Face):** https://huggingface.co/datasets/AkshaySasi/hypershadow
- **Trained models:** https://huggingface.co/AkshaySasi/hypershadow-models

Generate the data with one command. Train the detector in four minutes.
Or just load a tesseract's shadow and rotate it in a matplotlib window at
1 a.m., which I do not recommend for your sleep schedule but absolutely
recommend for your sense of wonder.

And if you beat my baselines, tell me. That is what benchmarks are for.

---

*Akshay Sasi is an AI/ML engineer. This is his first research paper. The
UFO, regrettably, was a camera artifact.*
