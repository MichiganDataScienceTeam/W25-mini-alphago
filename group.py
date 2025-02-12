from typing import Self

class Group:
    """
    Group of Go stones

    Args:
        intersections: set of indices of stones in the group
        borders: set of indices of intersections bordering the group
        liberties: subset of borders containing empty intersections
        group_type: numeric type of the stones in the group
    """


    def __init__(self, intersections: set[int], borders: set[int],
                 liberties: set[int], group_type: int) -> None:
        self.intersections = intersections
        self.borders = borders
        self.liberties = liberties
        self.group_type = group_type
    

    def __str__(self) -> str:
        """Return only intersections and liberties"""

        return f"Ints: {self.intersections} Libs: {self.liberties}"


    def __repr__(self) -> str:
        """Repr identical to str"""

        return str(self)
    

    def copy(self) -> Self:
        """Create copy of this group"""

        return Group(
            intersections = self.intersections.copy(),
            borders = self.borders.copy(),
            liberties = self.liberties.copy(),
            group_type = self.group_type
        )


    def union_in_place_(self, other: Self) -> Self:
        """
        Union this group with another group
        Changes only this group
        """

        if self.group_type != other.group_type:
            raise ValueError("Groups may only union with same type Groups")
        
        self.intersections |= other.intersections
        self.borders |= other.borders
        self.liberties |= other.liberties

        self.borders -= self.intersections
        self.liberties -= self.intersections

        if not self.liberties <= self.borders:
            raise NotImplementedError("Oops")
        
        return self
    

    @staticmethod
    def add_union(groups: list[Self], new_stone_group: Self) -> list[Self]:
        """
        Returns a minimized list of disjoint groups after adding a
        new group containing exactly 1 stone

        Args:
            groups: list of Groups
            new_stone_group: Group containing exactly 1 stone
        """

        if len(new_stone_group.intersections) != 1:
            raise ValueError("new_stone_group must have exactly one intersection")
        
        out = []
        need_union = [new_stone_group]
        
        new_stone = list(new_stone_group.intersections)[0]

        # Split groups
        for group in groups:
            if new_stone in group.borders and group.group_type == new_stone_group.group_type:
                need_union.append(group.copy())
            else:
                out.append(group.copy())
        
        # Union groups of same stone containing the new stone as a liberty
        new_intersections = set().union(*[group.intersections for group in need_union])
        new_borders = set().union(*[group.borders for group in need_union])
        new_liberties = set().union(*[group.liberties for group in need_union])

        new_borders -= new_intersections
        new_liberties -= new_intersections

        unioned = Group(
            intersections=new_intersections,
            borders=new_borders,
            liberties=new_liberties,
            group_type=new_stone_group.group_type
        )
        
        # Add unioned group back to group list
        out.append(unioned)
        
        # New stone cannot be liberty
        for group in out:
            group.liberties -= {new_stone}
        
        return out


    def replenish_liberties(self, opens: set[int]) -> set[int]:
        """
        Adds back liberties after intersections were captured
        Returns the resulting set of liberties

        Args:
            opens: set of newly open intersections
        """

        self.liberties |= self.borders & opens
        
        return self.liberties


    def trim_liberties(self, groups: list[Self]) -> set[int]:
        """
        Removes all invalid liberties in this Group based on
        intersections provided in a list of Groups in place
        Also returns the resulting set of liberties

        Args:
            groups: list of Groups containing stones
        """

        for group in groups:
            self.liberties -= group.intersections
        
        return self.liberties

