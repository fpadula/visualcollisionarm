def in_interval(from_a, to_a, angle):
    # Always read 'from' angle to 'to' angle in ccw direction    
    if(to_a < from_a):
        # this means that the interval crosses the 0 position, so we have to split the checking
        return in_interval(from_a, 360, angle) or in_interval(0, to_a, angle)
    else:
        return (angle >= from_a) and (angle <= to_a)
current_angle = 329
target_angle = 31
from_a_limit = 330
to_a_limit = 30
cw_delta = (current_angle - target_angle) % 360
ccw_delta = (target_angle - current_angle) % 360
print("Current angle: %.2f" % current_angle)
print("Target angle: %.2f" % target_angle)
print("Limits: from %.2f to %.2f" % (from_a_limit, to_a_limit))
print("Clockwise delta (%.2f)" % cw_delta)
print("Counter-clockwise delta (%.2f)" % ccw_delta)
if(cw_delta < ccw_delta):    
    to_a = current_angle
    from_a = target_angle
    direction_to_turn = "Clockwise"
    turn_delta = cw_delta
else:
    to_a = target_angle
    from_a = current_angle
    direction_to_turn = "Counter-clockwise"
    turn_delta = ccw_delta
violate_limits = in_interval(from_a, to_a, from_a_limit) or in_interval(from_a, to_a, to_a_limit)
print("Smallest delta violates joint limits: %s" % (violate_limits))
if(violate_limits):
    if(cw_delta < ccw_delta):
        direction_to_turn = "Counter-clockwise"
        turn_delta = ccw_delta
    else:
        direction_to_turn = "Clockwise"
        turn_delta = cw_delta
print("Turn direction: %s, delta: %.2f" % (direction_to_turn, turn_delta))