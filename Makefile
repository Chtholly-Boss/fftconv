EXT_DIR = ./ext
TEST_DIR = ./tests

all:
	@pip install ${EXT_DIR}
	@echo "Running All Tests"
	@pytest 

testf:
	@pip install ${EXT_DIR}
	@echo "Running FFT 1D Tests"
	@pytest $(TEST_DIR)/test_fft*.py

testc:
	@pip install ${EXT_DIR}
	@echo "Running Convolution 1D Tests"
	@pytest $(TEST_DIR)/test_conv*.py

clean:
	rm -rf __pycache__
	rm -rf $(TEST_DIR)/__pycache__
	rm -rf .pytest_cache

.PHONY: clean
