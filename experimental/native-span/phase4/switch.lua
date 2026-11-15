-- Switch slots mid-playback the way input.conf does (vf-command aji slot N), every 8 s of
-- media time (~190 frames at 23.976) counted from the first frame played.
local seq = {}
for s in (mp.get_opt("aji-seq") or ""):gmatch("[^.]+") do seq[#seq + 1] = s end
local first, last = nil, 0
mp.msg.warn("switch sequence: " .. table.concat(seq, " "))
mp.add_periodic_timer(0.02, function()
    local t = mp.get_property_number("time-pos")
    if not t then return end
    first = first or t
    local k = math.floor((t - first) / 8)
    if k > last and k <= #seq then
        last = k
        mp.msg.warn(string.format("SWITCH t=%.2f -> slot %s", t - first, seq[k]))
        mp.commandv("vf-command", "aji", "slot", seq[k])
    end
end)
