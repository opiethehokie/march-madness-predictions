#   Copyright 2016-2018 Michael Peters
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


from collections import OrderedDict


def simulate_tourney(teams_ids, predictions, tourney_format):

    def pairwise(it):
        it = iter(it)
        while True:
            yield next(it), next(it)

    def play_game(teama, seeda, teamb, seedb):
        if teams_ids[teama] < teams_ids[teamb]:
            matchup = "%d_%d" % (teams_ids[teama], teams_ids[teamb])
            prediction = predictions[matchup]
            winner = teama if prediction >= .5 else teamb
        else:
            matchup = "%d_%d" % (teams_ids[teamb], teams_ids[teama])
            prediction = predictions[matchup]
            winner = teamb if prediction >= .5 else teama
        print('%s %s vs %s %s = %s %f' % (seeda, teama, seedb, teamb, winner, prediction))
        return (seeda, winner) if winner == teama else (seedb, winner)

    def simulate(tformat, rnd):
        if len(tformat) > 1:
            print('\nROUND %d:' % rnd)
            winners = OrderedDict()
            if rnd == 0:
                for seed, team in tformat.items():
                    if '|' in team:
                        teama, teamb = team.split(' | ')
                        winners = tformat
                        (_, winner) = play_game(teama, seed, teamb, seed)
                        winners[seed] = winner
            else:
                for seeda, seedb in pairwise(tformat.keys()):
                    teama = tformat[seeda]
                    teamb = tformat[seedb]
                    wseed, winner = play_game(teama, seeda, teamb, seedb)
                    winners[wseed] = winner
            rnd += 1
            simulate(winners, rnd)

    simulate(tourney_format, 0)
