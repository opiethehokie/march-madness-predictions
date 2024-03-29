from collections import OrderedDict

import yaml

import yamlordereddictloader


def simulate_tourney(teams_ids, predictions, tourney_year):

    def pairwise(data):
        it = iter(data)
        try:
            while True:
                yield next(it), next(it)
        except StopIteration:
            return

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

    tfile = open('data/tourney_format_%s.yml' % tourney_year)
    simulate(yaml.load(tfile, Loader=yamlordereddictloader.Loader), 0)
