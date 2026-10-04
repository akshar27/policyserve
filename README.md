# PolicyServe

A real-time serving layer for robot control policies implementing the
dual-rate "System 2 / System 1" architecture used in production
vision-language-action (VLA) robotics stacks: a slow multimodal
reasoner decides *what to do next*, a fast local controller decides
*how to move right now*, and the two run decoupled so the fast loop
never stalls waiting on the slow one.

**Status:** in design.
