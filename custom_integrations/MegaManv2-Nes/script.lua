screens_offsets = {
    {x = 0, y = 8},
    {x = 1, y = 8},
    {x = 2, y = 8},
    {x = 3, y = 8},
    {x = 3, y = 7},
    {x = 3, y = 6},
    {x = 3, y = 5},
    {x = 3, y = 4},
    {x = 4, y = 4},
    {x = 5, y = 4},
    {x = 5, y = 3},
    {x = 5, y = 2},
    {x = 5, y = 1},
    {x = 5, y = 0},
    {x = 6, y = 0},
    {x = 7, y = 0},
    {x = 7, y = 1},
    {x = 7, y = 2},
    {x = 7, y = 3},
    {x = 8, y = 3},
    {x = 9, y = 3},
    {x = 10, y = 3},
    {x = 11, y = 3},
    {x = 12, y = 3}
}
screen_size = {x = 256, y = 240}
prev_distance = -1

function wavefront_expansion_reward()
    screen_offset = screens_offsets[data.screen+1]
    x = screen_size.x * screen_offset.x + data.x
    y = screen_size.y * screen_offset.y + data.y

    -- placeholder
    distance = x + y

    if prev_distance == -1 then
        distance_diff = 0
        prev_distance = distance
    else
        distance_diff = distance - prev_distance
        prev_distance = distance
    end

    return distance_diff
end