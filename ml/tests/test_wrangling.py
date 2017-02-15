#   Copyright 2016 Michael Peters
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


from ml import wrangling


def test_dervied_stats():
    season_stats = {'score': [5, 5], 'score-against': [5, 2],
                    'fgm': [5, 0], 'fgm-against': [5, 3],
                    'fga': [5, 77], 'fga-against': [5, 3]}
    assert len(wrangling.derive_stats(season_stats)) == 15

def test_descriptive_stats():
    season_stats = {'score': [5, 5]}
    stats = wrangling.describe_stats(season_stats)
    assert len(stats) == 5
    assert stats == [5, 5, 0.0, 5.0, 5.0]
