"""Narration for the HNMD explainer, Khan-Academy style.

Each scene: id, title, text.  Visual actions sync to timestamps of anchor
substrings inside the text (character-level alignment from ElevenLabs).
"""

SCENES = [
    dict(
        id="s01",
        title="Cold open",
        text=(
            "Let's say you've trained a neural network to predict the future. "
            "Not the next word in a sentence, but the next few hundred values of a real physical signal. "
            "The temperature of a transformer. The traffic on a freeway. "
            "Today I want to walk through a recent paper that asks a really nice question. "
            "What if the best way to improve a forecasting model isn't a better architecture, but a better definition of what counts as a good forecast? "
            "The paper is called Hierarchical NURBS-Inspired Multi-Domain Loss. That's a mouthful, so let's just call it H-N-M-D. "
            "And by the end of this video, every word in that name is going to make sense, because we're going to build the whole thing up from scratch."
        ),
    ),
    dict(
        id="s02",
        title="What is a time series?",
        text=(
            "So first, from the very beginning. What is a time series? It's just a list of measurements taken over time. "
            "Here, I'll draw one. Say we measure the oil temperature of an electricity transformer, every single hour. "
            "Each dot is one measurement, and if we connect them, we get a curve that wiggles through time. "
            "Now, forecasting is a fill-in-the-blank problem. "
            "We show the model a window of the recent past, say ninety-six steps. We call that the look-back window. "
            "And we ask it to draw the next chunk of the curve, the horizon, which might be another ninety-six steps, or even seven hundred and twenty. "
            "And usually we don't just have one signal. We have many channels recorded together. Temperature, load, humidity, whatever. "
            "So the input is really a little matrix, time along one axis, channels along the other. "
            "The model reads the green region, and it has to produce the orange region. That's the whole game."
        ),
    ),
    dict(
        id="s03",
        title="Training and MSE",
        text=(
            "Now, how do we train a model to do this? "
            "In this paper, the model itself is kept deliberately simple. A plain three-layer M-L-P. "
            "It takes the look-back window in, passes it through a few linear layers with activations, and spits out the entire forecast in one shot. "
            "And just like any network you've trained, learning happens by gradient descent on a loss function. "
            "The loss takes two things, the prediction and the ground truth, and boils their disagreement down to a single number. "
            "The classic choice is mean squared error. "
            "At every time step, measure the vertical gap between the predicted curve and the true curve. Square it. Average all those squares. "
            "Small number, good forecast. Big number, bad forecast. "
            "And this matters more than it looks, because the model becomes whatever its loss function rewards. "
            "The loss is the model's entire definition of truth."
        ),
    ),
    dict(
        id="s04",
        title="What MSE misses",
        text=(
            "So what could possibly be wrong with mean squared error? Let me show you. "
            "Here's a ground truth signal with a strong daily rhythm. Up every morning, down every night. "
            "Now let me draw two candidate forecasts. "
            "Candidate A is just a flat line through the middle. It has no rhythm, no shape, it commits to nothing. "
            "But notice, at every time step it's never catastrophically far from the truth. So its average squared error is actually pretty decent. "
            "Candidate B nails the shape. The peaks, the valleys, the frequency. But it's shifted in time, just slightly. "
            "Now every peak lines up with a valley, and point-wise, it pays full price at every single step. "
            "So mean squared error can literally prefer the flat, useless forecast over the one that understood the structure. "
            "And deep networks are opportunists. Studies of training dynamics show they follow the easiest path the loss allows. "
            "Train on a point-wise loss, and they drift toward exactly these blurry, over-smoothed averages. "
            "Now, people have attacked this before with shape-aware losses. DILATE. Tilde-Q. They build in invariance to small shifts and distortions. "
            "But this paper goes down a different road. Instead of comparing raw points, it first extracts the structure of both curves, at several scales, and then grades the structure itself."
        ),
    ),
    dict(
        id="s05",
        title="The roadmap",
        text=(
            "So here's the plan, in three ingredients. "
            "Ingredient one, a machine that takes any time series and turns it into structure. That machine is built out of splines. "
            "Ingredient two, a hierarchy. We'll run that machine several times, so the series splits into layers, from big slow trends down to fine fast wiggles. "
            "And ingredient three, a report card. Compare prediction and truth, layer by layer, in three different views. Where the curve is. How it's moving. And what rhythms it contains. "
            "That's the entire method. Now we just have to build each piece. And the first piece means we need to talk about splines."
        ),
    ),
    dict(
        id="s06",
        title="From polynomials to splines",
        text=(
            "Forget time series for a minute. Suppose you just have a handful of data points, and you want a smooth curve through them. "
            "The classic move is to fit a polynomial. And for gentle data, that works fine. "
            "But polynomials are global creatures. To follow lots of points, a single polynomial needs a high degree, and high-degree polynomials famously oscillate. "
            "You fix the fit over here, and the curve goes wild over there, because every coefficient affects the entire curve at once. "
            "So mathematicians borrowed a better idea from shipbuilders, who used to bend thin wooden strips, called splines, around pegs to draw smooth hull curves. "
            "The mathematical version goes like this. Chop the x-axis into intervals at chosen locations called knots. "
            "On each interval, use a simple low-degree polynomial, usually a cubic. "
            "And at every knot, force the neighboring pieces to agree. Same value, same slope, same curvature. The seams become invisible. "
            "One smooth, flexible curve, built from simple local pieces. That's a spline."
        ),
    ),
    dict(
        id="s07",
        title="B-spline basis functions",
        text=(
            "Now here's the elegant part, and it's the B in B-splines. Basis. "
            "Instead of juggling separate pieces, we build the whole spline out of standard building blocks, called basis functions. "
            "Watch how they're constructed. This is the Cox-de Boor recursion, and it's beautifully simple. "
            "Degree zero. Each knot interval gets a block function. One on its own interval, zero everywhere else. Rigid and boxy, but local. "
            "Now blend each block with its neighbor, ramping one down while the next ramps up. The blocks melt into little tents. That's degree one. "
            "Blend the tents with their neighbors, and you get smooth, rounded bumps. Degree two. "
            "Do it one more time, and you get cubic bumps. Degree three. Lovely smooth hills, and this is the standard choice. "
            "And here's the property that makes everything work. Each bump is non-zero only over a few neighboring intervals. This is called local support. "
            "Nudge one bump, and the curve changes only in that neighborhood. The wild global behavior of polynomials is gone."
        ),
    ),
    dict(
        id="s08",
        title="Fitting with a basis",
        text=(
            "So how does a pile of bumps become a curve that fits data? "
            "Give every bump a coefficient. Stretch each bump vertically by its coefficient, and add them all up. "
            "The result is a smooth curve, and by choosing the coefficients, you can bend it into almost any shape. "
            "And if you're coming from machine learning, notice what this is. It's just linear regression. The bumps are the features. "
            "Evaluate every basis function at every time step, collect the results into a matrix N, and solve ordinary least squares for the coefficients. "
            "C equals, N-transpose N, inverse, times N-transpose y. "
            "One matrix solve. No gradient descent, no iterations. "
            "And that speed matters, because this fit is about to run inside a loss function, on every single training batch."
        ),
    ),
    dict(
        id="s09",
        title="NURBS",
        text=(
            "One more upgrade and we arrive at NURBS. The name stands for Non-Uniform Rational B-Spline. "
            "Non-uniform just says the knots don't have to be evenly spaced. "
            "Rational is the interesting part. Every basis bump gets its own weight. "
            "Multiply each bump by its weight, then divide by the sum of all the weighted bumps, so at every position, everything still adds up to one. "
            "So what does a weight actually do? Watch. If I crank up the weight on one bump, its share grows, and the curve gets pulled toward that control point, like it suddenly has more gravity. "
            "Turn the weight down, and the curve barely listens to that point at all. "
            "This is the technology behind curves and surfaces in CAD and 3D modeling, where designers tune those weights by hand to sculpt exact shapes. "
            "In classic NURBS fitting, the weights are extra free parameters, and you optimize them. "
            "Hold that thought, because this paper does something sneakier with them."
        ),
    ),
    dict(
        id="s10",
        title="Weights from the data",
        text=(
            "Here's the twist. When this loss function fits a spline, it does not optimize the weights at all. "
            "It computes them, directly from the data, with a fixed weighting function. "
            "Slide a small window along the series, and inside each window, compute the variance. How turbulent is the signal right here? "
            "Flat, calm regions give low variance. Jumpy, stormy regions give high variance. "
            "Then push those variance scores through a softmax, and use the result as the NURBS weights. "
            "If you know how attention works in a transformer, this should feel very familiar. It's a softmax over regions of the sequence, deciding where the representation should focus. "
            "The effect is that the spline spends its flexibility on the turbulent regions, exactly where the interesting dynamics live. "
            "And the framework is general. You could condition on autocorrelation, or entropy. Variance is simply the criterion this paper uses."
        ),
    ),
    dict(
        id="s11",
        title="Complexity hierarchy",
        text=(
            "Now for the hierarchical part. "
            "A real signal is a pile of overlapping processes, and they have different complexities. "
            "Think about our transformer's temperature. A slow seasonal drift. A daily work cycle riding on top. Sharp load spikes on top of that. And then noise. "
            "Fit all of it with one spline, and everything gets smeared together. So instead, we peel the onion. "
            "Level one. Fit a spline with only a few knots. It physically cannot wiggle fast, so it captures only the slow, coarse structure. "
            "Now subtract that fit from the signal. What's left over is called the residual, everything the simple spline could not explain. "
            "Level two. Fit the residual with more knots, catching medium-scale structure. Subtract again. "
            "And keep going, each level more flexible than the last. "
            "If you've seen gradient boosting, this is the same soul. Every stage fits the leftovers of the stage before. "
            "Formally, the residual starts out as the signal itself. Each level fits its residual, then hands the remainder down. "
            "And the original signal equals the sum of all the levels, plus whatever tiny residual survives at the end."
        ),
    ),
    dict(
        id="s12",
        title="The full operator S",
        text=(
            "Let's assemble the full decomposition machine. The paper calls it S. "
            "We're handed a series. Set the residual equal to it. Then we loop, up to five levels in the experiments. "
            "Step one. Lay down this level's knots. And each level gets a denser grid than the last, so capacity grows as we descend. "
            "Step two. Compute this level's weights. Sliding-window variance on the current residual, pushed through the softmax. "
            "Step three. Build the weighted NURBS basis from those knots and weights. "
            "Step four. One least-squares solve gives the coefficients, and multiplying back out gives this level's component. "
            "Step five. Subtract the component, and carry the residual down to the next level. "
            "We stop at the level cap, or early, if the residual is already basically flat. "
            "Out comes a stack of curves, sorted from coarse to fine. And the implementation also keeps the sums of neighboring levels, so the transitions between scales get graded too."
        ),
    ),
    dict(
        id="s13",
        title="Three domains",
        text=(
            "And now, the loss itself. "
            "Take the model's prediction, and the ground truth. Run both of them through the exact same decomposition. "
            "So instead of comparing two raw curves, we compare them level by level. Structure against structure. "
            "And at every level, we look through three different lenses. "
            "Lens one, the time domain. Normalize the two curves and compare values, position by position. Are you in the right place? "
            "Lens two, the derivative domain. Take the difference between consecutive points, the discrete slope, and compare those. Are you moving the right way? This is the lens that catches shape. "
            "And lens three, the frequency domain. Quick primer, because this is the magic one. "
            "Any signal can be rewritten as a sum of sine waves of different frequencies. "
            "The fast Fourier transform, the F-F-T, tells you the strength of each ingredient. "
            "A strong daily cycle shows up as one bright spike at one cycle per day, no matter when the cycle starts. "
            "So comparing spectra asks, do you have the right rhythms? A slightly shifted forecast still keeps the right spectrum. Remember candidate B. "
            "Finally, we weight the three lenses with three dials. Alpha for position. Beta for motion. Gamma for rhythm. And add it all up. "
            "Those dials are tuned per dataset. And interestingly, the tuned settings usually turn alpha way down, and lean hard on motion and rhythm."
        ),
    ),
    dict(
        id="s14",
        title="Gradient weighting",
        text=(
            "There's one final trick, and it lives in the backward pass. "
            "During decomposition, we measure how wrong the prediction is at every level. Take the error, raise it to a power x, and average within each level. "
            "Normalize those level errors so they sum to one, and you get an importance score for every level of the hierarchy. "
            "Then, when gradients flow back through the decomposition into the model, each level's gradient gets scaled by its importance. "
            "Think about what that does. If the coarse trend is fine, but the daily rhythm is way off, then the rhythm level dominates the gradient, and this training step goes to fixing rhythm. "
            "It's like a teacher who grades every skill, but spends the red ink where you're weakest. And it re-decides that on every single batch."
        ),
    ),
    dict(
        id="s15",
        title="Results",
        text=(
            "So, does any of this actually help? Here's the experiment, and I love how clean it is. "
            "Fix one architecture, that plain three-layer M-L-P. Fix the optimizer, the look-back of ninety-six, and horizons from ninety-six out to seven hundred twenty steps ahead. "
            "Train the identical model three ways. With mean squared error. With Tilde-Q, the recent shape-aware loss. And with H-N-M-D. Then judge everyone with the same yardsticks, on held-out data. "
            "On the hourly transformer dataset E-T-T-h-one, averaged over all four horizons, M-S-E training lands at zero point four eight seven. "
            "Tilde-Q improves that to zero point four six one. "
            "And H-N-M-D reaches zero point four four two. "
            "On its sibling, E-T-T-h-two, the story repeats. Zero point five five seven, versus zero point four eight five, versus zero point four five seven. "
            "Same model. Same data. Roughly nine percent better than plain M-S-E training, purely from changing what the model was told to care about. "
            "And on the completed benchmark averages in the paper, H-N-M-D takes every single win. "
            "Now, to be fair. The gains are biggest on these hourly datasets. On electricity, traffic, and weather data, the margins over Tilde-Q get thinner. And a specialized architecture like PatchTST can still win some datasets outright. "
            "But look at what it does to the forecasts themselves. The M-S-E-trained model draws a cautious average through the middle. The H-N-M-D-trained model actually commits, to the dip, and to the recovery."
        ),
    ),
    dict(
        id="s16",
        title="Recap",
        text=(
            "Let's put the whole machine back together, end to end. "
            "The model makes a forecast. Prediction and truth both flow through the spline decomposition. "
            "Hierarchical, because the levels peel off structure from coarse to fine. "
            "NURBS-inspired, because variance-driven softmax weights tell every spline where to focus. "
            "And multi-domain, because each level is graded on position, on motion, and on rhythm. "
            "The level errors become importance scores. Importance shapes the gradients. And the gradients teach the model, batch after batch, to draw forecasts that have the structure of reality, not just low average error. "
            "To me, the deeper lesson is the one we started with. A loss function is a model's definition of truth. "
            "Define truth as matching points, and you get blurry averages. "
            "Define it as matching structure, at every scale, in every view, and the very same network learns to see shape. "
            "The authors point at what's next. Letting the hierarchy choose its own depth. Learning the weighting function instead of fixing it. "
            "But the core idea stands on its own. And now, every word in Hierarchical NURBS-Inspired Multi-Domain Loss is one you've built from scratch. "
            "Thanks for watching."
        ),
    ),
]

VOICE_ID = "cjVigY5qzO86Huf0OWal"  # Eric: smooth, trustworthy, middle-aged male
MODEL_ID = "eleven_multilingual_v2"
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
}
