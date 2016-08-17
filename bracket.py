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

import csv
import yaml
import yamlordereddictloader

from collections import OrderedDict


def team_id_mapping():
    with open('data/teams.csv') as csvfile:
        lines = csvfile.readlines()[1:]
        reader = csv.reader(lines)
        teams = { v:int(k) for k, v in reader }
    return teams

def tourney_format():
    # yaml format doesn't have an explicit ordering but intentionally keep order from file
    return yaml.load(open('data/tourney-2016.yml'), Loader=yamlordereddictloader.Loader)

def prediction_confidences():
    with open('submissions/submission.csv') as csvfile:
        lines = csvfile.readlines()[1:]
        reader = csv.reader(lines)
        predictions = { k[5:]:float(v) for k, v in reader }
    return predictions

teamid = team_id_mapping()
tourney = tourney_format()
predictions = prediction_confidences()

def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)

def log(f, msg):
    print msg
    f.write(msg + "\n")
        
with open('submissions/bracket.txt', 'w') as f:

    def simulate(teams, rnd):
        if len(teams) > 1:
            log(f, "\nROUND %d:" % rnd)
            winners = OrderedDict()
            if rnd == 0:
                for seed, team in teams.iteritems():
                    if '|' in team:
                        teama, teamb = team.split(' | ')
                        winners = teams
                        (_, winner) = play_game(teama, seed, teamb, seed)
                        winners[seed] = winner
            else:
                for seeda, seedb in pairwise(teams.keys()):
                    teama = teams[seeda]
                    teamb = teams[seedb]
                    wseed, winner = play_game(teama, seeda, teamb, seedb)
                    winners[wseed] = winner
            rnd += 1
            simulate(winners, rnd)
            
    def play_game(teama, seeda, teamb, seedb):
        teama_id = teamid[teama]
        teamb_id = teamid[teamb]
        if teama_id < teamb_id:
            matchup = "%d_%d" % (teama_id, teamb_id)
            prediction = predictions[matchup]
            winner = teama if prediction >= .5 else teamb
        else:
            matchup = "%d_%d" % (teamb_id, teama_id)
            prediction = predictions[matchup]
            winner = teamb if prediction >= .5 else teama
        log(f, "%s %s vs %s %s = %s %f" % (seeda, teama, seedb, teamb, winner, prediction))
        return (seeda, winner) if winner == teama else (seedb, winner)

    simulate(tourney, 0)
