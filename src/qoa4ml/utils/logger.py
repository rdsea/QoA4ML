import logging

qoa_logger = logging.getLogger(f"qoa4ml---{__name__}")

# Attach the default handler only once — double-imports / reloads must not
# cause duplicated log lines. ``propagate`` is left at the default (True)
# so callers using ``pytest caplog`` or a configured root-logger setup
# continue to see qoa4ml messages.
if not qoa_logger.handlers:
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(
        logging.Formatter(
            "%(module)s : %(asctime)s : %(levelname)s  - %(message)s",
        )
    )
    qoa_logger.addHandler(c_handler)
