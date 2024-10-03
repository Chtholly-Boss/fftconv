EXT_DIR = ./ext
TEST_DIR = ./tests

all:
	@echo "Installing cusfft..."
	@pip install ${EXT_DIR}
	@echo "Running All Tests"
	@pytest 

test_fft:
	@echo "Installing cusfft..."
	@pip install ${EXT_DIR}
	@echo "Running FFT 1D Tests"
	@pytest $(TEST_DIR)/test_fft*.py

test_conv:
	@echo "Installing cusfft..."
	@pip install ${EXT_DIR}
	@echo "Running Convolution 1D Tests"
	@pytest $(TEST_DIR)/test_conv*.py

clean:
	@echo "Cleaning Caches"
	rm -rf __pycache__
	rm -rf $(TEST_DIR)/__pycache__
	rm -rf .pytest_cache

.PHONY: clean
