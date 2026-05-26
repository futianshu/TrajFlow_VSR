# Related Work Plan

Related work should be organized by the problem chain, not by chronology.

## Blind And Real-World VSR

Use this section to motivate unknown degradation, artifact propagation, and
attention reliability. The comparison target is not only fidelity, but whether
the method avoids spreading unreliable evidence.

Required contrasts:

- BasicVSR / BasicVSR++.
- RealBasicVSR.
- RealViformer.

## Probabilistic Transport And Alignment

This section motivates why one hard optical flow field is insufficient under
occlusion, motion blur, and repeated texture.

Required contrasts:

- Optical-flow and deformable alignment.
- Attention-based soft matching.
- Entropic OT and unbalanced OT.
- Schrodinger Bridge image/video restoration.

## Long-Context Video Dynamics

This section explains why long VSR should not be treated as a generic token
sequence.

Required contrasts:

- Recurrent VSR.
- Transformer VSR.
- Mamba/SSM VSR.
- Koopman dynamics for lifted latent observables.

## Generative Restoration

This section separates high-frequency posterior generation from deterministic
low-frequency reconstruction.

Required contrasts:

- Diffusion VSR and robust restoration recipes.
- DiT/video restoration priors.
- Flow matching and rectified flow.
- Consistency distillation for low-step sampling.

## Arbitrary-Scale And Signal Representation

This section positions arbitrary scale as operator learning over a video
function, not just coordinate MLP interpolation.

Required contrasts:

- LIIF/INR-style image SR.
- Continuous video representation.
- Neural operators.
- Wavelet and anti-aliasing methods.

## Uncertainty, Reliability, And Hallucination Control

This section supports the claim that reliability is a calibration variable used
in transport, data consistency, and posterior strength.

Required evidence:

- Reliability-error correlation.
- Reliability ECE.
- Selective reconstruction curves.
- Failure cases where low reliability prevents unsafe high-frequency priors.
