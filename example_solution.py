from base_bot import BaseBot


class ExampleBot(BaseBot):
    def take_action(self, game_state) -> int:
        """
        Prosty bot, który skacze, gdy gracz jest blisko przeszkody.
        """
        player_y = game_state.get("player_y", 0)
        spikes = game_state.get("spikes", [])

        # Sprawdź najbliższą przeszkodę
        for spike in spikes:
            spike_x = spike.get("x", 0)
            spike_y = spike.get("y", 0)
            if spike_x < 100:  # Jeśli przeszkoda jest blisko
                if (
                    abs(player_y - spike_y) < 50
                ):  # Jeśli gracz jest na wysokości przeszkody
                    return 1  # Skocz

        return 0  # Nie rób nic

    def load_model(self, model_path: str = "model") -> object | None:
        pass
