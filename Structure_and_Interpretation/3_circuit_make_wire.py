# Dropping down into Python for this function because we need ...

def make_wire(name=None, self_probe=False):
    signal_value = False
    action_procedures = []

    def set_signal_bang(new_value):
        # ... nonlocal assignment (for which Eli Bendersky has a good
        # explanation:
        # http://eli.thegreenplace.net/2011/05/15/understanding-
        # unboundlocalerror-in-python);
        # we need plain Python for this because
        # https://github.com/hylang/hy/issues/246 is still open at
        # press time (see also https://github.com/hylang/hy/issues/664)
        nonlocal signal_value
        if signal_value != new_value:
            signal_value = new_value
            for procedure in action_procedures:
                procedure()
        else:
            return 'done'

    def accept_action_procedure_bang(procedure):
        action_procedures.append(procedure)
        procedure()

    def dispatch(m):
        if m == 'get_signal':
            return signal_value
        elif m == 'set_signal!':
            return set_signal_bang
        elif m == 'add_action!':
            return accept_action_procedure_bang
        else:
            raise ValueError("unknown operation")

    return dispatch
