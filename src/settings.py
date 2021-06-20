_DEFAULT_SETTINGS = {'add_unreachables': True, 'state_type': "GPSState", 'norm': "L2",
    'possible_actions': sorted([(1, 0), (0, 1), (-1, 0), (0, -1), (-1, 1), (1, -1), (1, 1), (-1, -1), (0,0)]),
    'determinism': .4, 'seed': None}


class Settings(object):
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            print('Creating the object')
            cls._instance = super(Settings, cls).__new__(cls)
            # Put any initialization here.
            settings_ = dict(_DEFAULT_SETTINGS)
            settings_.update(kwargs)
            for key in settings_:
                setattr(cls._instance, key, settings_[key])
        return cls._instance

    def reset(self):
        for key in _DEFAULT_SETTINGS:
            setattr(self, key, _DEFAULT_SETTINGS[key])
