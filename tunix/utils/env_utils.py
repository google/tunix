"""Utils for environment."""


def fs_open(filepath, mode='rb'):
  """Opens a file using fsspec."""
  try:
    from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile  # pylint: disable=g-import-not-at-top

    return gfile.Open(filepath, mode)
  except Exception:  # pylint: disable=broad-except
    import fsspec  # pylint: disable=g-import-not-at-top

    return fsspec.open(filepath, mode)
