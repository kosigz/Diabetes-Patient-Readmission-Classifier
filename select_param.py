# generates parameter values given a cost function by finding an upper bound on
# the parameter and minimizing the cost via binary search until the difference
# in cost values drops below the threshold
def select_param(costFn, initial=1, minimum=0, epsilon=0.1):
    infinity = float("Inf")
    curr, lower, upper, diff = float(initial), float(minimum), infinity, infinity

    # memoize costs from each parameter run
    _costs = {}
    def cost(val):
        try:
            return _costs[val]
        except KeyError:
            _costs[val] = costFn(val)

        return _costs[val]

    # start with the initial value as a guess
    yield curr

    cost(curr)
    last = curr
    while abs(diff) > epsilon:
        curr = min(curr * 2, (upper + lower) / 2)

        # once upper and lower bounds on the parameter value have been
        # established, track the difference between the costs at each endpoint
        if lower != infinity and upper != infinity:
            diff = cost(upper) - cost(lower)

        if curr == last:
            if diff > 0:
                curr = (last + upper) / 2
            else:
                curr = (last + lower) / 2

        yield curr

        if curr < last:
            # guess lies between lower bound and last guess
            if cost(curr) > cost(last):
                # guess cost lies between cost at lower bound and last guess
                # guess becomes new lower bound
                lower = curr
            else:
                # guess cost is lower than lower bound and last guess
                # last guess becomes new upper bound
                upper, last = last, curr
        elif curr > last:
            # guess lies between last guess and upper bound
            if cost(curr) > cost(last):
                # guess cost lies between cost at last guess and upper bound
                # guess becomes new upper bound
                upper = curr
            else:
                # guess cost is lower than last guess and upper bound
                # last guess becomes new upper bound
                lower, last = last, curr

    return
