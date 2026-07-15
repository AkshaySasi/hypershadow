# I Taught an AI to Recognize the Shadows of Four-Dimensional Objects

*How a blurry Pentagon UFO video led me to build a benchmark, a detector, and an arXiv paper, on a laptop with a 4 GB GPU.*

---

## It started with a weird video

A few years ago the Pentagon released a batch of UAP footage. One clip showed
a strange star-shaped object drifting across the night sky, and the internet
did what the internet does: aliens, obviously.

My first thought was different. I thought about shadows.

Hold a cube in front of a lamp and its shadow on the wall is a hexagon, or a
square, or something in between, depending on how you turn it. The shadow is
genuinely two-dimensional, but it is not an honest 2D object. It is a 3D
object wearing a 2D disguise. So I wondered: if something four-dimensional
ever moved through our world, we would not see it as it is. We would see its
3D shadow. And it would look, for lack of a better word, weird.

(For the record: that star-shaped object was later convincingly explained as
bokeh, an out-of-focus light source filtered through a triangular camera
aperture. The mundane explanation won, as it almost always does. Remember
this, it matters later.)

The question that stuck with me was not "was that a UFO". It was: **if a
higher-dimensional object cast a shadow into our world, could we even tell?
Is a 4D shadow measurably different from an ordinary 3D thing?**

Nobody had ever tested this. So I did.

## Turning a shower thought into an experiment

Here is the problem with questions like this: you cannot go collect 4D
objects. But you can do the next best thing, which is what physics has always
done with untestable-sounding questions: build a controlled laboratory where
you know the ground truth, and ask whether the difference is detectable
*in principle*.

Computers do not care how many dimensions an array has. A point in 4D space
is just four numbers. So I built a generator that creates:

- **Ordinary 3D shapes**: spheres, cubes, donuts, cylinders, a knotted tube.
- **Genuinely four, five and six dimensional shapes**: hyperspheres,
  tesseracts, Clifford tori, objects most people have only seen in YouTube
  animations.

Then I rotated each higher-dimensional object randomly in its own space and
projected it down to 3D, exactly the way a lamp projects a cube onto a wall.
The result is a dataset of 10,800 point clouds: half of them honest 3D
objects, half of them shadows of something bigger.

And because I did not want to fool myself, I spent most of my design effort
on ways the experiment could cheat:

- Every cloud is centred, rescaled and resampled to the same point count, so
  size and position carry no information.
- The ordinary 3D shapes get the same random rotations and the same noise,
  occlusion and sensor corruption as the shadows.
- The 3D class includes a solid ball, because the shadow of a 4D hypersphere
  is also a solid ball. "It fills a volume" is not allowed to be the answer.

That last point deserves a pause. The shadow of a 4D sphere IS a 3D ball.
Identical silhouette. The only difference is *where the mass concentrates
inside*. If you want to catch it, you need to notice density, not shape.

## What the experiments found

I ran four kinds of detectors on this dataset. The results surprised me
twice.

**Surprise one: the obvious tool fails.** The textbook approach for "what
dimension is my data" is intrinsic-dimension estimation (methods like TwoNN
and maximum-likelihood estimators). They scored about 72 to 74 percent,
barely better than a coin flip with a thumb on the scale. And in hindsight
the reason is beautiful: a shadow is still 3D data. Its dimension is not the
clue. The clue is what projection *does* to density and topology on the way
down.

**A small neural network does much better.** A deliberately tiny PointNet,
190k parameters, four minutes of training on my GTX 1650 Ti, reached 96.2
percent accuracy. More interestingly, when I held out entire shape families
during training, it still detected them at 79 to 91 percent. It had not
memorised my shape catalogue. It had learned what projection itself looks
like.

**Surprise two: the best detector has zero parameters.** This is my favourite
result in the paper. Watch an object *move* instead of looking at a still
frame. A rigid 3D object that rotates can always be aligned frame-to-frame
by a rigid 3D transform, that is essentially the definition of rigid. But
the shadow of a rigidly rotating 4D object morphs. No rigid 3D motion can
explain it. Measure the leftover "impossible deformation" with the Kabsch
algorithm, a closed-form formula from the 1970s, and you get a single number
that separates the two classes with AUROC 0.982. No training. No neural
network. One threshold.

Motion leaks dimensional information that still images hide. Humans, by the
way, seem to sense a version of this too: a 2023 vision-science study found
that people in VR can distinguish rigid from non-rigid motion of a
hypercube's projection. Now there is a machine baseline to compare against.

## What this does not prove (and why that is the point)

Let me be very clear, because this topic attracts wishful thinking like a
porch light attracts moths: **none of this says higher dimensions exist.**
Everything was tested on simulated geometry where I knew the right answer.

What it does establish is narrower and, I think, more interesting:

1. Higher-dimensional shadows carry detectable statistical fingerprints,
   even under noise, occlusion and sensor-like corruption.
2. The standard dimensionality tools cannot see those fingerprints, and
   learned or motion-based methods can.
3. There exist what I call *dimensional witnesses*: numbers you can compute
   from ordinary 3D observations that are provably near zero under any rigid
   3D explanation. A large value does not tell you what the data is. It
   tells you what the data is *not*, and "not explainable by any rigid 3D
   process" is a statement with teeth.

That third idea is the one I care about long-term. It has the same logical
shape as a Bell inequality: you never observe the hidden thing directly, you
rule out the entire class of explanations that lack it. Bell did not
photograph hidden variables. He found a number whose violation made a whole
worldview untenable. The rigidity witness is a toy-scale cousin of that
logic, applied to spatial dimensions, validated in a lab where the truth is
known.

If genuinely multi-sensor, properly tracked anomaly data ever becomes
public, tools like this are how you would analyse it honestly: not "is it
aliens" but "is any rigid 3D explanation consistent with this trajectory".
And when the answer comes back "yes, a bird explains it fine", that is the
tool working, not failing. Remember the bokeh.

## The part nobody tells you about doing research alone

Some honest numbers, because I wish someone had told me this before I
started:

- Total hardware: one laptop, GTX 1650 Ti, 4 GB of VRAM, 8 GB of RAM. The
  full training run takes four minutes. Every experiment in the paper
  reproduces overnight on hardware most gamers would call outdated.
- The dataset generator is plain NumPy. No GPU needed at all.
- The most valuable hours were not spent training models. They were spent
  designing ways my own experiment could lie to me, and closing them one by
  one. The fairness rules in the paper took longer than the neural network.
- The scariest moment was finding a 2023 paper studying almost the same
  question in humans. It turned out to be the best thing that happened to
  the project: it proved the question was scientifically respectable, and
  it gave my machine results a human baseline to stand next to.

I genuinely believe there are hundreds of questions like this one sitting in
plain sight: too strange for a grant proposal, too small for a lab, exactly
right for one stubborn person with a consumer GPU.

## Try it yourself

Everything is public and reproducible from seeds:

- **Paper (arXiv):** [ARXIV_LINK]
- **Code and tests:** https://github.com/AkshaySasi/hypershadow
- **Dataset (Hugging Face):** https://huggingface.co/datasets/AkshaySasi/hypershadow
- **Trained models:** https://huggingface.co/AkshaySasi/hypershadow-models

Generate the data with one command, train the detector in four minutes, or
just load the point clouds and look at them. The shadow of a tesseract is a
genuinely strange thing to rotate in a matplotlib window at 1 a.m., and I
say that with affection.

If you beat my baselines, tell me. That is what benchmarks are for.

---

*Akshay Sasi is an AI/ML engineer. This is his first research paper. The
UFO, regrettably, was a camera artifact.*
