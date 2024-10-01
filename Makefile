all:
	echo "all"
test:
	@echo "Installing cusfft..."
	@pip install ./ext
	@echo "Running tests"
	@pytest
clean:
	echo "clean"

.PHONY: clean
