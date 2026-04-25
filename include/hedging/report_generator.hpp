#pragma once

#include "hedging_runner.hpp"
#include "../bs/black_scholes.hpp"
#include <string>

namespace hedging {

// Generates a comprehensive JSON report containing model inputs, 
// pricing results, Greeks, backtest settings, and attribution summaries.
void generate_json_report(
    const std::string& filepath,
    const HedgingMCConfig& config,
    const bs::BlackScholes& bs_initial,
    const HedgingMCSummary& summary
);

} // namespace hedging
