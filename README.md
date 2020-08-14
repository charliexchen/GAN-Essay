# Cambridge Maths Part III Essay -- Generative Adversarial Networks
For Cambridge Maths Part III, there was an option to write an essay reviewing several papers as a substitute for an exam. One of the essay titles available was on GANs, and I ended up taking this since I realised it was the best opportunity dive into deep neural networks within that course. 

(Sadly Cambridge courses had a tendency to be a little too theoretical for its own good, a fact which is often treated as a selling point for the Maths Tripos)

The [essay](https://github.com/charliexchen/GANEssay/blob/master/GAN%20Essay/GANEssay.pdf) was written from a mathematical perspective and reviewed the original papers on GANs by Ian Goodfellow, in addition to several other results hosted in the blog "[Off the Convex Path](https://www.offconvex.org/)".
However, for demonstration purposes, I also coded up some GANs which can learn simple distributions and MNIST, based on some online tutorials. 

Since the essay's target audience was supposed to be at the level of fellow Part III students (who doesn't have ML experience), it first covers some basics:
 * Quick overview of DNNs, CNNs, backprop etc.
 * Definition of GANs and basic results based on Goodfellow's paper.
 * Some quick demonstrations with code from this repo.

This is followed by some more interesting results:
 * Some exploration of theoretical limitations and practical challenges, along with some methods of mitigating those challenges which are used in practice.
 * A methodology for verifying the generalisation of GANs beyond its training data by leveraging a heuristic based on birthday attacks -- an area which had already been explored by computer scientists due to its utility in finding hash collisions.
 * Introducing the notion of generalisation of distances between probability distributions. This challenges some of the assumptions in Goodfellow's paper and provides some theoretical reasons for why GANs might generalise poorly or suffer from mode collapse.
 * Proof existence of (ε-approximate mixed) Nash Equilibriums in GANs under certain conditions. When people are first introduced to GANs they're often told that "eventually the discriminator will return 0.5 all the time, and the generator's outputs will hopefully be indistinguishable from the training data". This result essentially formalises this concept and shows that it is indeed a stable equilibrium which the GAN might arrive at.

<figure class="image" text-align="center">
  <img src="https://github.com/charliexchen/GANEssay/blob/master/GAN%20Essay/MNIST.png?raw=true">
  <figcaption text-align="center">Outputs of DCGAN trained on MNIST compared to training data (Right) </figcaption>
</figure>



This repo is a fork of the original repository which hosted the code anonymously -- The essays had to be submitted without any identifiers to avoid examiner bias. 
Since that is no longer an issue, I am migrating that code over to my named GitHub alongside with document containing my essay for posterity. Furthermore, there have been some more novel developments over at Off the Convex Path, and this code will allow me to start experimenting with that.

References may be found within the essay.



