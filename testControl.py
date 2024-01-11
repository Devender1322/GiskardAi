import giskard

# Load your model (replace with your specific code)
model = load_your_model()

# Define custom test function (can handle multiple inputs and outputs)
def my_custom_test(inputs, expected_outputs):
    # Ensure inputs and outputs have matching lengths
    assert len(inputs) == len(expected_outputs)

    # Iterate through each input-output pair
    for input_text, expected_output in zip(inputs, expected_outputs):
        prediction = model.predict(input_text)
        assert prediction == expected_output, f"Model prediction '{prediction}' did not match expected output '{expected_output}' for input '{input_text}'."

# Create test suite
test_suite = giskard.TestSuite()

# Define multiple sets of test data
test_inputs1 = ["Question 1", "Prompt 1"]
expected_outputs1 = ["Answer 1", "Response 1"]

test_inputs2 = ["Question 2", "Prompt 2"]
expected_outputs2 = ["Answer 2", "Response 2"]

# Add tests with different data
test_suite.add_test(my_custom_test, inputs=test_inputs1, expected_outputs=expected_outputs1, name="Test Set 1")
test_suite.add_test(my_custom_test, inputs=test_inputs2, expected_outputs=expected_outputs2, name="Test Set 2")

# Run the suite
results = test_suite.run(model)

# Print results
print(results)