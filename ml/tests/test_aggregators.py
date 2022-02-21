from ml import aggregators


def test_modified_rpi():
    season_stats = {1: {'opponents': [2, 3], 'results': [1, 1]}, 2: {'opponents': [1, 3], 'results': [0, 0]},
                    3: {'opponents': [1, 2], 'results': [0, 1]}}
    assert aggregators._opponents_win_percent(season_stats, [2, 3]) == .25
    assert aggregators._opponents_opponents_win_percent(season_stats, [2, 3]) == .625
    assert aggregators._rpi(season_stats, 1, [.25, .5, .25]) == .25 * 1 + .5 * .25 + .25 * .625
