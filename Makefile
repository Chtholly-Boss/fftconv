EXT_DIR = ./ext
TEST_DIR = ./tests

all:
	@echo "Installing cusfft..."
	@pip install ${EXT_DIR}
	@echo "Running All Tests"
	@pytest 

test1:
	@echo "Installing cusfft..."
	@pip install ${EXT_DIR}
	@echo "Running FFT 1D Tests"
	@pytest $(TEST_DIR)/test_fft_1d.py

test2:
	@echo "Installing cusfft..."
	@pip install ${EXT_DIR}
	@echo "Running FFT 2D Tests"
	@pytest $(TEST_DIR)/test_fft_2d.py

clean:
	@echo "Cleaning Caches"
	rm -rf __pycache__
	rm -rf $(TEST_DIR)/__pycache__
	rm -rf .pytest_cache

.PHONY: clean
