from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class MyPlugin(SupervisedPlugin):
    """
    Implemented your plugin (if any) here.
    """

    def __init__(self, ):
        """
        :param
        """
        super().__init__()

    def before_backward(self, strategy, **kwargs):
        """
        Example callback: before backward.
        """
        # This callback does nothing ...
        pass
