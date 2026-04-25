#include "../../include/hedging/report_generator.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace hedging {

void generate_json_report(
    const std::string& filepath,
    const HedgingMCConfig& config,
    const bs::BlackScholes& bs_initial,
    const HedgingMCSummary& summary) {
        
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filepath << " for writing JSON report.\n";
        return;
    }

    // Set precise formatting for floating point numbers
    out << std::setprecision(6) << std::fixed;

    out << "{\n";
    
    // 1. Model Inputs
    out << "  \"model_inputs\": {\n";
    out << "    \"initial_spot\": " << config.S0 << ",\n";
    out << "    \"strike\": " << config.K << ",\n";
    out << "    \"risk_free_rate\": " << config.r << ",\n";
    out << "    \"implied_volatility\": " << config.sigma_hedge << ",\n";
    out << "    \"maturity_years\": " << config.T << ",\n";
    out << "    \"option_type\": \"" << (config.is_call ? "Call" : "Put") << "\"\n";
    out << "  },\n";

    // 2. Pricing Results
    out << "  \"pricing_results\": {\n";
    out << "    \"option_premium\": " << (config.is_call ? bs_initial.call_price() : bs_initial.put_price()) << ",\n";
    out << "    \"d1\": " << bs_initial.d1() << ",\n";
    out << "    \"d2\": " << bs_initial.d2() << "\n";
    out << "  },\n";

    // 3. Initial Greeks
    out << "  \"initial_greeks\": {\n";
    out << "    \"delta\": " << (config.is_call ? bs_initial.call_delta() : bs_initial.put_delta()) << ",\n";
    out << "    \"gamma\": " << bs_initial.gamma() << ",\n";
    out << "    \"theta\": " << (config.is_call ? bs_initial.call_theta() : bs_initial.put_theta()) << ",\n";
    out << "    \"vega\": " << bs_initial.vega() << ",\n";
    out << "    \"rho\": " << (config.is_call ? bs_initial.call_rho() : bs_initial.put_rho()) << "\n";
    out << "  },\n";

    // 4. Backtest Settings
    out << "  \"backtest_settings\": {\n";
    out << "    \"realized_drift_mu\": " << config.mu << ",\n";
    out << "    \"realized_vol_sigma\": " << config.sigma_sim << ",\n";
    out << "    \"num_paths\": " << config.num_paths << ",\n";
    out << "    \"num_rebalancing_steps\": " << config.num_steps << ",\n";
    out << "    \"dt\": " << (config.T / config.num_steps) << ",\n";
    out << "    \"transaction_cost_per_unit\": " << config.cost_per_unit << ",\n";
    out << "    \"random_seed\": " << config.seed << "\n";
    out << "  },\n";

    // 5. P&L Summary
    out << "  \"pnl_summary\": {\n";
    out << "    \"mean_pnl\": " << summary.mean_pnl << ",\n";
    out << "    \"std_pnl\": " << summary.std_pnl << ",\n";
    out << "    \"min_pnl\": " << summary.min_pnl << ",\n";
    out << "    \"max_pnl\": " << summary.max_pnl << "\n";
    out << "  },\n";

    // 6. Attribution Summary
    out << "  \"attribution_summary\": {\n";
    out << "    \"mean_theta_carry\": " << summary.mean_theta << ",\n";
    out << "    \"mean_gamma_exposure\": " << summary.mean_gamma << ",\n";
    out << "    \"mean_financing\": " << summary.mean_financing << ",\n";
    out << "    \"mean_transaction_costs\": " << summary.mean_txn_cost << ",\n";
    out << "    \"mean_residual_error\": " << summary.mean_residual << "\n";
    out << "  }\n";

    out << "}\n";
    out.close();
}

} // namespace hedging
