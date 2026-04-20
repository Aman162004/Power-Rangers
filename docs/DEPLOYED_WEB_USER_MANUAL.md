# Power-Rangers Quick User Manual

Delhi Power Demand Predictor  
For the current deployed web dashboard

## 1. What This System Does

Power-Rangers helps users view Delhi electricity demand forecasts through a protected web dashboard. It shows forecasted load, actual load when available, uncertainty bands, forecast accuracy values, and the predicted peak demand time.

The current web dashboard supports:

- Login and logout.
- Registration using an invite token.
- Forecast date selection.
- Temperature scenario adjustment from -5 C to +5 C.
- Forecast refresh.
- P10, P50, and P90 forecast lines.
- Actual demand line when backend data is available.
- Latest actual, latest P50, MAE, MAPE, and peak forecast.
- Chart hover values for exact timestamp checks.

The current dashboard does not include a web admin panel, report export button, weather overlay chart, automatic alert notifications, or web-based model training.

## 2. User Types

### Operator

Uses the dashboard during operations to check demand forecast, actuals, uncertainty, and peak timing.

### Energy Analyst

Reviews forecast behavior, compares scenarios, and reports unusual model or data behavior.

### Administrator

Manages access through backend-supported admin APIs. The current web dashboard does not include a visible admin management screen.

### ML/Data/DevOps Teams

Maintain the model, data pipelines, backend, frontend, deployment, and troubleshooting outside the main dashboard UI.

## 3. Open the Web App

1. Open your browser.
2. Enter the deployed Power-Rangers web URL.
3. Wait for the page to load.
4. If you are not signed in, the Sign In page appears.
5. If your session is still valid, the dashboard opens.

Use Chrome, Edge, Firefox, or Safari. If your organization requires VPN, connect to VPN before opening the page.

## 4. Register With Invite Token

Use this only if you do not already have an account.

1. Open the Sign In page.
2. Select `Register with invite token`.
3. Enter a username with at least 3 characters.
4. Enter your email address.
5. Enter your full name if needed.
6. Select `Continue`.
7. Enter a password with at least 6 characters.
8. Re-enter the same password.
9. Paste the invite token given by the administrator.
10. Select `Register`.
11. After successful registration, the dashboard opens.

Common registration problems:

- Username is too short.
- Email is not valid.
- Password is too short.
- Passwords do not match.
- Invite token is missing, expired, or invalid.
- Username or email already exists.

## 5. Sign In

1. Open the Power-Rangers web page.
2. Enter your username.
3. Enter your password.
4. Select `Sign In`.
5. If login succeeds, the dashboard opens.
6. If login fails, check the username and password and try again.

If login still fails, contact the administrator.

## 6. Use the Dashboard

After login, the dashboard contains these sections:

- Overview
- Architecture
- SDLC Timeline
- Analytics
- Capabilities
- Deployment

Use the left sidebar to move between sections. For daily use, go directly to `Analytics`.

## 7. Run a Forecast Check

1. Sign in.
2. Select `Analytics`.
3. Check the `Forecast date`.
4. Keep `Temperature adjustment` at `0 C` for the base case.
5. Select `Refresh`.
6. Wait for the chart and values to update.
7. Review latest actual, latest P50, MAE, MAPE, and peak forecast.
8. Hover over the chart for exact timestamp values.

The current dashboard does not allow selecting a date later than the browser's current date.

## 8. Change the Temperature Scenario

Use this when you want a simple hotter or cooler load case.

1. Open `Analytics`.
2. Move the `Temperature adjustment` slider.
3. Use `0 C` for base case.
4. Move right for hotter cases.
5. Move left for cooler cases.
6. Select `Refresh` if the chart does not update automatically.

Current range:

- Minimum: `-5 C`
- Maximum: `+5 C`
- Step: `1 C`

Backend behavior:

- Each 1 C changes the forecast by 2 percent.
- `+5 C` means a +10 percent demand scenario.
- `-5 C` means a -10 percent demand scenario.

This is a simple load-scaling scenario, not a full weather simulation.

## 9. Read the Forecast Chart

Chart items:

- Actual line: Observed demand, shown only when backend actuals are available.
- P50: Main median forecast.
- P10: Lower forecast estimate.
- P90: Upper forecast estimate.
- P10-P90 band: Forecast uncertainty range.

How to read it:

1. Start with P50 for the main expected demand.
2. Use P90 for conservative planning.
3. Use P10 as the lower estimate.
4. Watch whether actuals are above or below P50.
5. Treat a wider P10-P90 band as higher uncertainty.
6. Use `Peak forecast` for the predicted maximum load and time.

## 10. Read the Values

- `Latest actual`: Latest observed demand in the returned data. Shows `--` if unavailable.
- `Latest P50`: Latest median forecast value in the current chart.
- `MAE`: Average absolute forecast error where actuals are available.
- `MAPE`: Average percentage forecast error where actuals are available.
- `Peak forecast`: Highest predicted demand and its timestamp.

Note: The backend computes RMSE, but the current dashboard does not show an RMSE card.

## 11. Operator Quick Workflow

1. Sign in.
2. Go to `Analytics`.
3. Set forecast date.
4. Keep temperature at `0 C`.
5. Select `Refresh`.
6. Read latest actual and latest P50.
7. Check MAE and MAPE.
8. Check P10-P90 uncertainty.
9. Read peak forecast value and time.
10. Test hotter or cooler scenario if needed.
11. Share the values using your team's normal process.
12. Select `Logout` when done.

## 12. Analyst Quick Workflow

1. Sign in.
2. Go to `Analytics`.
3. Select the date to review.
4. Refresh the base case at `0 C`.
5. Compare actuals with P50.
6. Check whether actuals fall outside P10-P90.
7. Compare base case with hotter or cooler scenarios.
8. Report unusual values to the ML or data team.

The current web dashboard does not export reports or CSV files.

## 13. Administrator Notes

The current web dashboard does not include a working admin screen. Admin actions exist through backend APIs:

- Create invite token: `POST /api/auth/admin/invite`
- List users: `GET /api/auth/admin/users`
- Update user role: `PUT /api/auth/admin/users/{user_id}/role`
- Deactivate user: `DELETE /api/auth/admin/users/{user_id}`

New registered users become `OPERATOR` by default.

## 14. Troubleshooting

### Page Does Not Open

1. Check the URL.
2. Refresh the browser.
3. Check internet or VPN.
4. Try another supported browser.
5. Contact support if it still fails.

### Login Fails

1. Re-enter username and password.
2. Check capitalization.
3. Try again.
4. Contact the administrator if it still fails.

### Forecast Does Not Load

If the dashboard shows `Unable to fetch forecast for the selected date`:

1. Check the selected date.
2. Set temperature adjustment back to `0 C`.
3. Select `Refresh`.
4. If it still fails, report the date and a screenshot.

Possible causes include backend downtime, unreachable forecast API, missing SLDC data, model inference failure, or missing model/config files.

### Actuals Are Missing

1. Confirm the selected date should have actual data.
2. Select `Refresh`.
3. Try a nearby date.
4. Report the missing date range to the data team.

### Forecast Looks Wrong

1. Confirm forecast date.
2. Confirm temperature setting.
3. Refresh.
4. Take a screenshot.
5. Report the issue with date, scenario value, and observed problem.

## 15. Glossary

- Actual: Observed load value.
- Forecast date: Date selected for forecast generation.
- MAE: Mean Absolute Error.
- MAPE: Mean Absolute Percentage Error.
- MW: Megawatt.
- P10: Lower forecast estimate.
- P50: Median forecast estimate.
- P90: Upper forecast estimate.
- Peak forecast: Highest predicted load and time.
- SLDC: State Load Dispatch Centre.
- TFT: Temporal Fusion Transformer.
- Uncertainty band: Area between P10 and P90.

