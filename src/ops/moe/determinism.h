#pragma once

namespace ninfer::ops {

// Startup-fixed switch: when set, MoE dispatch routes the expert combine through the
// order-deterministic contributions + reduce path at every token count. Function-local
// static keeps one instance across TUs; set once during engine construction.
[[nodiscard]] inline bool& moe_deterministic() {
    static bool value = false;
    return value;
}

} // namespace ninfer::ops
