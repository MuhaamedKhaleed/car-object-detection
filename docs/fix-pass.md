# Ordinary single-image inference, 25 November 2026

`car-box predict IMAGE.png MODEL/model.keras` loads an explicit trusted local Keras artifact and returns one xyxy box in the original image pixel dimensions. Use `car-box --help` for the installed entrypoint. Inference uses saved input dimensions and the same RGB resize/divide255 preprocessing as package training. Invalid/nonfinite/out-of-bounds/reversed predictions fail instead of being sorted or clipped into plausible boxes.

The four-output model still predicts one box only. It has no confidence or class head and cannot express multiple cars. No multiple-object or no-car correctness is claimed. Original .h5 checkpoints are not bundled or newly recovered. Tests use controlled constant-output Keras fixtures to verify exact pixel rescaling, CLI and reversed-corner rejection, not model quality.
