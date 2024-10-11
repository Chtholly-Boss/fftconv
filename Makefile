EXT_DIR = ./ext
TEST_DIR = ./tests
CLEAN_DIRS = . $(TEST_DIR) $(EXT_DIR)

all:
	@echo "Running All Tests"
	@pytest 

ext:
	@pip install ${EXT_DIR}

tprecision:
	@echo "Checking Precision" 
	@pytest $(TEST_DIR)/test_precision*.py

tspeed:
	@echo "Checking Speed"
	@pytest $(TEST_DIR)/test_speed*.py

clean:
	rm -rf $(addsuffix /__pycache__, $(CLEAN_DIRS))
	rm -rf $(addsuffix /.pytest_cache, $(CLEAN_DIRS))
	rm -rf $(EXT_DIR)/build
	rm -rf $(EXT_DIR)/*.egg-info

.PHONY: clean
