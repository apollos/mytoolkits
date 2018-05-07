import sys
import operator

# Cost is basically: was there a match or not.
# The other numbers are cumulative costs and matches.

def lowest_cost_action(ic, dc, sc, im, dm, sm, cost):
    """Given the following values, choose the action (insertion, deletion,
    or substitution), that results in the lowest cost (ties are broken using
    the 'match' score).  This is used within the dynamic programming algorithm.

    * ic - insertion cost

    * dc - deletion cost

    * sc - substitution cost

    * im - insertion match (score)

    * dm - deletion match (score)

    * sm - substitution match (score)
    """
    best_action = None
    best_match_count = -1
    min_cost = min(ic, dc, sc)
    if min_cost == sc and cost == 0:
        best_action = 'equal'
        best_match_count = sm
    elif min_cost == sc and cost == 1:
        best_action = 'replace'
        best_match_count = sm
    elif min_cost == ic and im > best_match_count:
        best_action = 'insert'
        best_match_count = im
    elif min_cost == dc and dm > best_match_count:
        best_action = 'delete'
        best_match_count = dm
    return best_action


def edit_distance(seq1, seq2, action_function=lowest_cost_action, test=operator.eq):
    """Computes the edit distance between the two given sequences.
    This uses the relatively fast method that only constructs
    two columns of the 2d array for edits.  This function actually uses four columns
    because we track the number of matches too.
    """
    m = len(seq1)
    n = len(seq2)
    # Special, easy cases:
    if seq1 == seq2:
        return 0, n
    if m == 0:
        return n, 0
    if n == 0:
        return m, 0
    v0 = [0] * (n + 1)     # The two 'error' columns
    v1 = [0] * (n + 1)
    m0 = [0] * (n + 1)     # The two 'match' columns
    m1 = [0] * (n + 1)
    for i in range(1, n + 1):
        v0[i] = i
    for i in range(1, m + 1):
        v1[0] = i
        for j in range(1, n + 1):
            cost = 0 if test(seq1[i - 1], seq2[j - 1]) else 1
            # The costs
            ins_cost = v1[j - 1] + 1
            del_cost = v0[j] + 1
            sub_cost = v0[j - 1] + cost
            # Match counts
            ins_match = m1[j - 1]
            del_match = m0[j]
            sub_match = m0[j - 1] + int(not cost)

            action = action_function(ins_cost, del_cost, sub_cost, ins_match,
                                     del_match, sub_match, cost)

            if action in ['equal', 'replace']:
                v1[j] = sub_cost
                m1[j] = sub_match
            elif action == 'insert':
                v1[j] = ins_cost
                m1[j] = ins_match
            elif action == 'delete':
                v1[j] = del_cost
                m1[j] = del_match
            else:
                raise Exception('Invalid dynamic programming option returned!')
                # Copy the columns over
        for i in range(0, n + 1):
            v0[i] = v1[i]
            m0[i] = m1[i]
    print(v1)
    print(m1)
    return v1[n], m1[n]


def main():
    """Read two files line-by-line and print edit distances between each pair
    of lines. Will terminate at the end of the shorter of the two files."""

    if len(sys.argv) != 3:
        print('Usage: {} <file1> <file2>'.format(sys.argv[0]))
        exit(-1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    with open(file1) as f1, open(file2) as f2:
        for line1, line2 in zip(f1, f2):
            print("Line 1: {}".format(line1.strip()))
            print("Line 2: {}".format(line2.strip()))

            dist, matches = edit_distance(line1.split(), line2.split())
            print('Distance: {}'.format(dist))
            print('=' * 80)
            print('Matches: {}'.format(matches))


if __name__ == "__main__":
    main()
