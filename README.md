# project11-neural-ode

Neural Network from Scratch for ODE Solving

Team 11, Course 35-380. A Rust implementation of a discrete state-transition
surrogate for a damped pendulum. The neural network and its gradients are
implemented manually; no automatic differentiation or ML framework is used.

Requires stable Rust with the platform's native compiler/linker.

The physical model is theta' = omega and
omega' = -(g / length) sin(theta) - damping * omega.
Default parameters are g = 9.81, length = 1, damping = 0.1.

Names, student IDs, individual contributions and the course deadline are not
provided. Complete these using the team's real records before submission.
