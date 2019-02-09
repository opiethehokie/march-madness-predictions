#   Copyright 2016-2019 Michael Peters
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from ml2 import aggregators


def test_modified_rpi():
    season_stats = {1: {'opponents': [2, 3], 'results': [1, 1]}, 2: {'opponents': [1, 3], 'results': [0, 0]},
                    3: {'opponents': [1, 2], 'results': [0, 1]}}
    assert aggregators._opponents_win_percent(season_stats, [2, 3]) == .25
    assert aggregators._opponents_opponents_win_percent(season_stats, [2, 3]) == .625
    assert aggregators._rpi(season_stats, 1, [.25, .5, .25]) == .25 * 1 + .5 * .25 + .25 * .625
