import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("simple_xor.onnx")
test_input = np.array([[0, 1]], dtype=np.float32)
outputs = session.run(None, {'my_input': test_input})
prediction = outputs[0][0][0]
print(f"--- ONNX Inference Result ---")
print(f"Input: [0, 1]")
print(f"Model Prediction: {prediction:.4f}")
print(f"Final Decision: {1 if prediction > 0.5 else 0}")