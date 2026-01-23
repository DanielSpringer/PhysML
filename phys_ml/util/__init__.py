
def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except:
        return False
