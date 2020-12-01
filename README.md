# pypatippet
Phase (and tempo) Inference from Point Process Event Timing - python3 implementation. 

See Jonathan Cannon's reference implementation in MATLAB: https://github.com/joncannon/PIPPET

Notes:
- While PIPPET (phase inference alone) isn't explicitly implemented here, tempo inference can be 'switched off' by setting `xbar0=1.0` and `sigma_tempo=0.0`. 
- Currently only a single event stream is supported. It should be trivial to extend this to multiple event streams.

Please feel free to raise a pull request, additional utilities would be much appreciated.

**Warning:** this has _not_ been thoroughly tested, use at your own risk!
