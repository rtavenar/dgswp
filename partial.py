import numpy as np
import warnings
from sortedcontainers import SortedList
from kneed import KneeLocator


class PartialOT1d:
    def __init__(self, max_iter) -> None:
        self.max_iter = max_iter

    @property
    def x_sorted(self):
       return self.x[self.indices_sort_x]

    @property
    def y_sorted(self):
       return self.y[self.indices_sort_y]

    @property
    def n_x(self):
       return len(self.x)

    @property
    def n_y(self):
       return len(self.y)

    def preprocess(self, x, y):
        """Given two 1d distributions `x` and `y`:
        
        1. `self.indices_sort_x` sorts `x` (ie. `x[self.indices_sort_x]` is sorted) and
           `self.indices_sort_y` sorts `y` (ie. `y[self.indices_sort_y]` is sorted)
        
        2. stack them into a single distrib such that:
        
        * the new distrib is sorted with sort indices (wrt a stack of sorted x and sorted y) `self.indices_sort_xy`
        * `self.sorted_distrib_indicator` is a vector of zeros and ones where 0 means 
          "this point comes from x" and 1 means "this point comes from y"
        """
        self.x = x if type(x) is np.ndarray else np.array(x)
        self.y = y if type(y) is np.ndarray else np.array(y)

        self.indices_sort_x = np.argsort(x)
        self.indices_sort_y = np.argsort(y)

        xy = np.concatenate((self.x_sorted, self.y_sorted))
        self.indices_sort_xy = np.argsort(xy)

        idx = np.concatenate((np.zeros(self.n_x, dtype=int), np.ones(self.n_y, dtype=int)))
        self.sorted_distrib_indicator = idx[self.indices_sort_xy]

    def _insert_constant_values(self, arr, distrib_index):
        """Takes `arr` as input. For each position `i` in `arr`, 
        if `self.sorted_distrib_indicator[i]==distrib_index`,
        the value from `arr` is copied, 
        otherwise, the previous value that was copied from `arr` is repeated.

        Examples
        --------
        >>> p = PartialOT1d(-1)
        >>> x = [0, 3, 4]
        >>> y = [1, 2, 5]
        >>> p.preprocess(x, y)
        >>> p._insert_constant_values([1, -1, -1, 2, 3, -1], 0)
        array([1, 1, 1, 2, 3, 3])
        """
        arr_insert = []
        for i in range(len(arr)):
            if self.sorted_distrib_indicator[i]==distrib_index:
                arr_insert.append(arr[i])
            elif i == 0:
                arr_insert.append(0)
            else:
                arr_insert.append(arr_insert[i-1])
        return np.array(arr_insert)

    def compute_cumulative_sum_differences(self):
        """Computes difference between cumulative sums for both distribs.

        The cumulative sum vector for a sorted x is:

            cumsum_x = [x_0, x_0 + x_1, ..., x_0 + ... + x_n]

        This vector is then extend to reach a length of 2*n by repeating 
        values at places that correspond to an y item.
        In other words, if the order of x and y elements on the real 
        line is something like x-y-y-x..., then the extended vector is 
        (note the repetitions):

            cumsum_x = [x_0, x_0, x_0, x_0 + x_1, ..., x_0 + ... + x_n]

        Overall, this function returns `cumsum_x - cumsum_y` where `cumsum_x`
        and `cumsum_y` are the extended versions.

        Examples
        --------
        >>> p = PartialOT1d(-1)
        >>> x = [-1, 3, 4]
        >>> y = [1, 2, 5]
        >>> p.preprocess(x, y)
        >>> p.sorted_distrib_indicator
        array([0, 1, 1, 0, 0, 1])
        >>> p.compute_cumulative_sum_differences()  # [-1, -1, -1, 2, 6, 6] - [0, 1, 3, 3, 3, 8]
        array([-1, -2, -4, -1,  3, -2])
        """
        cum_sum_xs = np.cumsum(self.x_sorted)
        cum_sum_ys = np.cumsum(self.y_sorted)
        cum_sum = np.concatenate((cum_sum_xs, cum_sum_ys))
        cum_sum = cum_sum[self.indices_sort_xy]

        cum_sum_x = self._insert_constant_values(cum_sum, 0)
        cum_sum_y = self._insert_constant_values(cum_sum, 1)

        return cum_sum_x - cum_sum_y

    def compute_rank_differences(self):
        """Precompute important rank-related quantities for better group generation.
        
        Three quantities are returned:

        * `ranks_xy` is an array that gathers ranks of the elements in 
          their original distrib, eg. if the distrib indicator is
          [0, 1, 1, 0, 0, 1], then `rank_xy` will be:
          [0, 0, 1, 1, 2, 2]
        * `diff_ranks` is computed from `ranks_xy_x_cum` and `ranks_xy_y_cum`.
          For the example above, we would have:
          
            ranks_xy_x_cum = [1, 1, 1, 2, 3, 3]
            ranks_xy_y_cum = [0, 1, 2, 2, 2, 3]

          And `diff_ranks` is just `ranks_xy_x_cum - ranks_xy_y_cum`.

        Examples
        --------
        >>> p = PartialOT1d(max_iter=2)
        >>> x = [-2, 2, 3]
        >>> y = [-1, 1, 5]
        >>> p.preprocess(x, y)
        >>> ranks_xy, diff_ranks = p.compute_rank_differences()
        >>> ranks_xy
        array([0, 0, 1, 1, 2, 2])
        >>> diff_ranks
        array([ 1,  0, -1,  0,  1,  0])
        """
        ranks_x, ranks_y = np.arange(self.n_x), np.arange(self.n_y)
        ranks_xy = np.concatenate((ranks_x, ranks_y))
        ranks_xy = ranks_xy[self.indices_sort_xy]

        ranks_xy_x = ranks_xy.copy()
        ranks_xy_x[self.sorted_distrib_indicator==1] = 0
        ranks_xy_x[self.sorted_distrib_indicator==0] += 1
        ranks_xy_x_cum = self._insert_constant_values(ranks_xy_x, 0)

        ranks_xy_y = ranks_xy.copy()
        ranks_xy_y[self.sorted_distrib_indicator==0] = 0
        ranks_xy_y[self.sorted_distrib_indicator==1] += 1
        ranks_xy_y_cum = self._insert_constant_values(ranks_xy_y, 1)

        diff_ranks = ranks_xy_x_cum - ranks_xy_y_cum

        return ranks_xy, diff_ranks
    
    def compute_costs(self, diff_cum_sum, diff_ranks):
        """For each element in sorted `x`, compute its group (cf note below).
        Then compute the cost for each group and sort all groups in increasing 
        cost order.

        Note: the "group" of a point x_i in x is the minimal set of adjacent 
        points (starting at x_i and extending to the right) that one should 
        take to get a balanced set (ie. a set in which we have as many 
        elements from x as elements from y)

        Examples
        --------
        >>> p = PartialOT1d(max_iter=3)
        >>> x = [1, 2, 5, 6]
        >>> y = [3, 4, 11, 12]
        >>> p.preprocess(x, y)
        >>> diff_cum_sum = p.compute_cumulative_sum_differences()
        >>> ranks_xy, diff_ranks = p.compute_rank_differences()
        >>> costs, pack_costs_cumsum = p.compute_costs(diff_cum_sum, diff_ranks)
        >>> list(costs)
        [(1, 2, 1), (3, 4, 1), (5, 6, 5)]
        """
        l_costs = []
        self._group_starting_at = {}
        self._group_ending_at = {}
        last_pos_for_rank_x = {}
        last_pos_for_rank_y = {}
        for i in range(len(diff_ranks)):
            # For each item in either distrib, find the scope of the smallest
            # "group" that would start at that point and extend on the right, 
            # if one exists, and store the cost of this "group" by relying 
            # on differences of cumulative sums
            idx_end = i
            cur_rank = diff_ranks[i]
            if self.sorted_distrib_indicator[i] == 0:
                target_rank = cur_rank - 1
                idx_start = last_pos_for_rank_y.get(target_rank, None)
                last_pos_for_rank_x[cur_rank] = i
            else:
                target_rank = cur_rank + 1
                idx_start = last_pos_for_rank_x.get(target_rank, None)
                last_pos_for_rank_y[cur_rank] = i
            if idx_start is not None:
                if idx_start == 0:
                    cost = diff_cum_sum[idx_end] # - 0
                else:
                    cost = diff_cum_sum[idx_end]   - diff_cum_sum[idx_start - 1]
                # i: start of the "group", "next_pos": end of the "group", abs(cost): cost of the group
                if idx_end == idx_start + 1:
                    l_costs.append((idx_start, idx_end, abs(cost)))
                self._group_starting_at[idx_start] = {"ends_at": idx_end, "cost": abs(cost)}
                assert idx_end not in self._group_ending_at
                self._group_ending_at[idx_end] = {"starts_at": idx_start, "cost": abs(cost)}
        return SortedList(l_costs, key=lambda x: x[2]), self.precompute_pack_costs_cumsum()
    
    def precompute_pack_costs_cumsum(self):
        """For each position `i` at which a pack could end,
        Compute (using dynamic programming and the costs of groups that have been precomputed)
        the cost of the largest pack ending at `i`.

        This is useful because this can be used later, to compute the cost of any pack in O(1)
        (cf. `_compute_cost_for_pack`).
        """
        pack_costs_cumsum = {}
        for i in range(self.n_x + self.n_y):
            # If there is a group ending there
            # (if not, this cannot be the end of a pack)
            if i in self._group_ending_at:
                start = self._group_ending_at[i]["starts_at"]
                additional_cost = self._group_ending_at[i]["cost"]
                pack_costs_cumsum[i] = pack_costs_cumsum.get(start - 1, 0) + additional_cost
        return pack_costs_cumsum

     
    @classmethod
    def _insert_new_pack(cls, packs: SortedList, candidate_pack):
        """Insert the `candidate_pack` into the sorted list of `packs`.
        `packs` is modified in-place and the pack in which the candidate 
        is inserted is returned.

        Examples
        --------
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [5, 7])
        [2, 9]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 9]], key=<function <lambda> at ...>)
        >>> 
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [11, 12])
        [11, 12]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 4], [8, 9], [11, 12]], key=<function <lambda> at ...>)
        >>> 
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [5, 6])
        [2, 6]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 6], [8, 9]], key=<function <lambda> at ...>)
        >>> 
        >>> packs = SortedList([[2, 4], [8, 9]], key=lambda t: t[0])
        >>> PartialOT1d._insert_new_pack(packs, [6, 7])
        [6, 9]
        >>> packs # doctest: +ELLIPSIS
        SortedKeyList([[2, 4], [6, 9]], key=<function <lambda> at ...>)
        """
        i, j = candidate_pack
        idx = packs.bisect_left(candidate_pack)
        # Is `i` adjacent to `packs[idx - 1]` 
        # or `j` adjacent to `packs[idx]`?
        if idx > 0 and packs[idx - 1][1] == i - 1:
            # Extend the pack up to `j`
            p = packs.pop(idx - 1)
            candidate_pack = [p[0], candidate_pack[1]]
        idx = packs.bisect_left(candidate_pack)
        if idx < len(packs) and packs[idx][0] == j + 1:
            # Extend the pack from `i` on
            p = packs.pop(idx)
            candidate_pack = [candidate_pack[0], p[1]]
        packs.add(candidate_pack)
        return candidate_pack
    
    def _compute_cost_for_pack(self, idx_start, idx_end, pack_costs_cumsum):
        """Compute the associated cost for a pack (set of contiguous points
        included in the solution) ranging from `idx_start` to `idx_end` (both included).
        """
        return pack_costs_cumsum[idx_end] - pack_costs_cumsum.get(idx_start - 1, 0)
    
    def get_group_starting_at(self, i):
        return self._group_starting_at.get(i, {"ends_at": None, "cost": None})

    def generate_solution_using_marginal_costs(self, costs: SortedList, ranks_xy, pack_costs_cumsum):
        """Generate a solution from a sorted list of group costs.
        See the note in `compute_costs` docs for a definition of groups.

        The solution is a pair of lists. The first list contains the indices from `sorted_x`
        that are in the active set, and the second one contains the indices from `sorted_y`
        that are in the active set.

        Examples
        --------
        >>> p = PartialOT1d(max_iter=3)
        >>> x = [1, 2, 5, 6]
        >>> y = [3, 4, 11, 12]
        >>> p.preprocess(x, y)
        >>> diff_cum_sum = p.compute_cumulative_sum_differences()
        >>> ranks_xy, diff_ranks = p.compute_rank_differences()
        >>> costs, pack_costs_cumsum = p.compute_costs(diff_cum_sum, diff_ranks)
        >>> ranks_xy = p.compute_rank_differences()[0]
        >>> p.generate_solution_using_marginal_costs(costs, ranks_xy, pack_costs_cumsum)
        (array([1, 2, 3]), array([0, 1, 2]), [1, 1, 5])
        """
        max_iter = self.max_iter if self.max_iter != "elbow" else min(self.n_x, self.n_y)
        active_set = set()
        packs = SortedList(key=lambda t: t[0])
        list_marginal_costs = []
        list_active_set_inserts = []
        while len(costs) > 0 and max_iter > len(active_set) // 2:
            i, j, c = costs.pop(index=0)
            if i in active_set or j in active_set:
                continue
            new_pack = None
            # Case 1: j == i + 1 => "Simple" insert
            if j == i + 1:
                new_pack = PartialOT1d._insert_new_pack(packs, [i, j])
            # Case 2: insert a group that contains a pack
            elif [i + 1, j - 1] in packs:
                packs.remove([i + 1, j - 1])
                new_pack = PartialOT1d._insert_new_pack(packs, [i, j])
            # There should be no "Case 3"
            else:
                self._print_current_status(active_set, i, j)
                raise ValueError
            active_set.update({i, j})
            list_active_set_inserts.append({i, j})
            list_marginal_costs.append(c)
            
            # We now need to update the groups wrt the pack we have just created
            p_s, p_e = new_pack
            if p_s == 0 or p_e == self.n_x + self.n_y - 1:
                continue
            if self.sorted_distrib_indicator[p_s - 1] != self.sorted_distrib_indicator[p_e + 1]:
                # If (p_s - 1, p_s) and (p_e, p_e + 1) are groups, remove them
                if (self.get_group_starting_at(p_s - 1)["ends_at"] == p_s 
                        and self.get_group_starting_at(p_e)["ends_at"] == p_e + 1):
                    # Below we use discard instead of remove (discard does not throw an error
                    # if the item is not in the list) since it could be that the group 
                    # has already been removed beforehand because of an overlap with
                    # another pack
                    costs.discard((p_s - 1, p_s, self.get_group_starting_at(p_s - 1)["cost"]))
                    costs.discard((p_e, p_e + 1, self.get_group_starting_at(p_e)["cost"]))

                # Insert (p_s - 1, p_e + 1) as a new pseudo-group with marginal cost
                marginal_cost = (self._compute_cost_for_pack(p_s - 1, p_e + 1, pack_costs_cumsum)
                                 - self._compute_cost_for_pack(p_s, p_e, pack_costs_cumsum))
                costs.add((p_s - 1, p_e + 1, marginal_cost))

        # Generate index arrays in the order of insertion in the active set
        indices_sorted_x = np.array([ranks_xy[i] 
                                     if self.sorted_distrib_indicator[i] == 0 else ranks_xy[j] 
                                     for i, j in list_active_set_inserts])
        indices_sorted_y = np.array([ranks_xy[i] 
                                     if self.sorted_distrib_indicator[i] == 1 else ranks_xy[j] 
                                     for i, j in list_active_set_inserts])

        return (indices_sorted_x, indices_sorted_y, list_marginal_costs)
    
    def check_solution_valid(self, indices_x, indices_y):
        """Check that a solution (given by two lists of indices in 
        sorted x and sorted y respectively) is valid.

        Examples
        --------
        >>> p = PartialOT1d(-1)
        >>> x = [-2, 3, 5]
        >>> y = [-1, 1, 3]
        >>> p.preprocess(x, y)
        >>> p.check_solution_valid([0, 1], [0, 2])
        >>> p.check_solution_valid([0, 1], [0, 1, 2])  # This one raises a warning
        >>> p.check_solution_valid([0, 1], [0, 3])     # This one too
        """
        if len(indices_x) != len(indices_y):
            warnings.warn("A valid solution should have as many x's as y's", RuntimeWarning)
        if not np.all(np.array(indices_x) < self.n_x) or not np.all(np.array(indices_x) >= 0):
            warnings.warn(f"All x indices should be between 0 and {self.n_x - 1}", RuntimeWarning)
        if not np.all(np.array(indices_y) < self.n_y) or not np.all(np.array(indices_y) >= 0):
            warnings.warn(f"All y indices should be between 0 and {self.n_y - 1}", RuntimeWarning)
        # We could implement other checks here

    def fit(self, x, y):
        """Main method for the class.
        
        Does:
        
        1. Preprocessing of the distribs (sorted & co)
        2. Precomputations (ranks, cumulative sums)
        3. Extraction of groups
        4. Generate and return solution

        Note that the indices in `indices_x` and `indices_y` are ordered wrt their order of
        appearance in the solution such that `indices_x[:10]` (resp y) is the set of indices
        from x (resp. y) for the partial problem of size 10.

        Arguments
        ---------
        x : array-like of shape (n, )
            First distrib to be considered (weights are considered uniform)
        y : array-like of shape (m, )
            Second distrib to be considered (weights are considered uniform)

        Returns
        -------
        indices_x : np.ndarray of shape (self.max_iter, ) or (min(n, m), )
            Indices of elements from the x distribution to be included in the partial solutions
            Order of appearance in this array indicates order of inclusion in the solution
        indices_y : np.ndarray of shape (self.max_iter, ) or (min(n, m), )
            Indices of elements from the x distribution to be included in the partial solutions
            Order of appearance in this array indicates order of inclusion in the solution
        list_marginal_costs : list of length self.max_iter or min(n, m)
            List of marginal costs associated to the intermediate partial problems
            `np.cumsum(list_marginal_costs)` gives the corresponding total costs for intermediate partial problems

        Examples
        --------
        >>> p = PartialOT1d(max_iter=2)
        >>> x = [5, -2, 4]
        >>> y = [-1, 1, 3]
        >>> p.fit(x, y)
        (array([1, 2]), array([0, 2]), [1, 1])
        """
        # Sort distribs and keep track of their original indices (stored in instance attributes)
        self.preprocess(x, y)

        # Precompute useful quantities
        diff_cum_sum = self.compute_cumulative_sum_differences()
        ranks_xy, diff_ranks = self.compute_rank_differences()

        # Compute costs for "groups"
        costs, pack_costs_cumsum = self.compute_costs(diff_cum_sum, diff_ranks)

        # Generate solution from sorted costs
        sol_indices_x_sorted, sol_indices_y_sorted, sol_costs = self.generate_solution_using_marginal_costs(costs, ranks_xy, pack_costs_cumsum)

        self.check_solution_valid(sol_indices_x_sorted, sol_indices_y_sorted)

        # Convert back into indices in original `x` and `y` distribs
        self.indices_x_ = self.indices_sort_x[sol_indices_x_sorted]
        self.indices_y_ = self.indices_sort_y[sol_indices_y_sorted]
        self.marginal_costs_ = sol_costs

        if self.max_iter == "elbow":
            kneedle = KneeLocator(x=np.arange(len(self.marginal_costs_)), 
                                  y=np.cumsum(self.marginal_costs_), 
                                  S=1.0, 
                                  curve="convex", 
                                  direction="increasing")
            idx_elbow = int(kneedle.elbow)
            return (self.indices_x_[:idx_elbow + 1], 
                    self.indices_y_[:idx_elbow + 1], 
                    self.marginal_costs_[:idx_elbow + 1])
        else:
            return self.indices_x_, self.indices_y_, self.marginal_costs_

    def _print_current_status(self, active_set, i, j):
        print("=" * (15 + self.n_x + self.n_y) + "\nCurrent status")
        s = "Distribs:      "
        for v in self.sorted_distrib_indicator:
            if v == 0:
                s += "o"
            else:
                s += "-"
        print(s)
        s = "Active set:    "
        for pos in range(self.n_x + self.n_y):
            if pos in active_set:
                s += "x"
            else:
                s += " "
        print(s)
        s = "Current group: "
        for pos in range(self.n_x + self.n_y):
            if pos in [i, j]:
                s += "^"
            else:
                s += " "
        print(s)
        print("=" * (15 + self.n_x + self.n_y))

if __name__ == "__main__":
    pb = PartialOT1d(max_iter="elbow")
    np.random.seed(0)
    x = np.random.rand(30, )
    y = np.random.rand(40, )
    indices_x, indices_y, marginal_costs = pb.fit(x, y)
    print(len(indices_x), len(indices_y))
    print(indices_x)
    print(indices_y)
