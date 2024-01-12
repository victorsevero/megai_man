import os

import retro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integrations")
    )
    print(
        "MegaManv2-Nes"
        in retro.data.list_games(inttype=retro.data.Integrations.ALL)
    )
    env = retro.make("MegaManv2-Nes", inttype=retro.data.Integrations.ALL)
    print(env)


if __name__ == "__main__":
    main()
