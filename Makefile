PLAY = neural_mate_play
PRETRAIN = neural_mate_pretrain
TRAIN = neural_mate_train
DIAGNOSE = neural_mate_diagnose

ALL_BINS = $(PLAY) $(PRETRAIN) $(TRAIN) $(DIAGNOSE)

.PHONY: all re clean fclean

all: $(ALL_BINS)

$(PLAY):
	@echo '#!/usr/bin/env python' > $(PLAY)
	@echo 'import sys, os' >> $(PLAY)
	@echo 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))' >> $(PLAY)
	@echo 'from src.play import main' >> $(PLAY)
	@echo 'sys.exit(main() or 0)' >> $(PLAY)
	@chmod +x $(PLAY)
	@echo "Created $(PLAY)"

$(PRETRAIN):
	@echo '#!/usr/bin/env python' > $(PRETRAIN)
	@echo 'import sys, os' >> $(PRETRAIN)
	@echo 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))' >> $(PRETRAIN)
	@echo 'from src.pretraining.pretrain import main' >> $(PRETRAIN)
	@echo 'sys.exit(main() or 0)' >> $(PRETRAIN)
	@chmod +x $(PRETRAIN)
	@echo "Created $(PRETRAIN)"

$(TRAIN):
	@echo '#!/usr/bin/env python' > $(TRAIN)
	@echo 'import sys, os' >> $(TRAIN)
	@echo 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))' >> $(TRAIN)
	@echo 'from src.alphazero.train import main' >> $(TRAIN)
	@echo 'sys.exit(main() or 0)' >> $(TRAIN)
	@chmod +x $(TRAIN)
	@echo "Created $(TRAIN)"

$(DIAGNOSE):
	@echo '#!/usr/bin/env python' > $(DIAGNOSE)
	@echo 'import sys, os' >> $(DIAGNOSE)
	@echo 'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))' >> $(DIAGNOSE)
	@echo 'from diagnose_network import main' >> $(DIAGNOSE)
	@echo 'sys.exit(main() or 0)' >> $(DIAGNOSE)
	@chmod +x $(DIAGNOSE)
	@echo "Created $(DIAGNOSE)"

re: fclean all

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned cache files"

fclean: clean
	@rm -f $(ALL_BINS)
	@echo "Removed executables"
