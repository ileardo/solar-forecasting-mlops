"""
Minimal monitoring flow for solar forecasting system health checks.

This module provides a basic Prefect flow for monitoring model drift trends,
system health, and generating simple monitoring reports.
"""

from datetime import datetime, timedelta
from typing import Any, Dict

from prefect import flow, task
from prefect.logging import get_run_logger

from src.monitoring.db_writer import MonitoringDBWriter


@task
def analyze_drift_trends_task(days: int = 7) -> Dict[str, Any]:
    """
    Analyze drift trends over the last N days.

    Args:
        days: Number of days to analyze.

    Returns:
        Dict[str, Any]: Drift trend analysis results.
    """
    task_logger = get_run_logger()
    task_logger.info(f"Analyzing drift trends for last {days} days")

    try:
        db_writer = MonitoringDBWriter()

        # Simple SQL query to get drift statistics
        with db_writer._get_connection() as conn:
            with conn.cursor() as cursor:
                # Get drift counts by day
                cursor.execute(
                    """
                    SELECT
                        DATE(timestamp) as drift_date,
                        COUNT(*) as total_features,
                        SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drifted_features
                    FROM monitoring.data_drift
                    WHERE timestamp >= %s
                    GROUP BY DATE(timestamp)
                    ORDER BY drift_date DESC
                """,
                    [datetime.now() - timedelta(days=days)],
                )

                drift_by_day = cursor.fetchall()

                # Get most frequently drifted features
                cursor.execute(
                    """
                    SELECT
                        feature_name,
                        COUNT(*) as drift_count,
                        AVG(drift_score) as avg_drift_score
                    FROM monitoring.data_drift
                    WHERE timestamp >= %s AND drift_detected = true
                    GROUP BY feature_name
                    ORDER BY drift_count DESC
                    LIMIT 5
                """,
                    [datetime.now() - timedelta(days=days)],
                )

                top_drifted_features = cursor.fetchall()

        # Calculate summary statistics
        total_drift_events = sum(row[2] for row in drift_by_day)
        total_feature_checks = sum(row[1] for row in drift_by_day)
        drift_rate = (
            (total_drift_events / total_feature_checks * 100)
            if total_feature_checks > 0
            else 0
        )

        drift_analysis = {
            "analysis_period_days": days,
            "total_drift_events": total_drift_events,
            "total_feature_checks": total_feature_checks,
            "overall_drift_rate": round(drift_rate, 2),
            "drift_by_day": [
                {
                    "date": str(row[0]),
                    "total_features": row[1],
                    "drifted_features": row[2],
                    "drift_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0,
                }
                for row in drift_by_day
            ],
            "top_drifted_features": [
                {
                    "feature_name": row[0],
                    "drift_count": row[1],
                    "avg_drift_score": round(float(row[2]), 3),
                }
                for row in top_drifted_features
            ],
        }

        task_logger.info(
            f"Drift analysis complete: {drift_rate:.1f}% overall drift rate"
        )
        return drift_analysis

    except Exception as e:
        task_logger.error(f"Drift analysis failed: {str(e)}")
        return {"analysis_failed": True, "error": str(e)}


@task
def system_health_check_task() -> Dict[str, Any]:
    """
    Perform basic system health checks.

    Returns:
        Dict[str, Any]: System health status.
    """
    task_logger = get_run_logger()
    task_logger.info("Performing system health check")

    try:
        db_writer = MonitoringDBWriter()
        health_status = db_writer.health_check()

        # Additional checks
        with db_writer._get_connection() as conn:
            with conn.cursor() as cursor:
                # Check recent predictions
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM predictions
                    WHERE created_at >= %s
                """,
                    [datetime.now() - timedelta(days=1)],
                )
                recent_predictions = cursor.fetchone()[0]

                # Check reference data availability
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM monitoring.reference_data
                """
                )
                reference_data_count = cursor.fetchone()[0]

        system_health = {
            "database_healthy": health_status["healthy"],
            "recent_predictions_24h": recent_predictions,
            "reference_data_available": reference_data_count > 0,
            "reference_data_count": reference_data_count,
            "monitoring_tables_accessible": True,
            "check_timestamp": datetime.now().isoformat(),
        }

        # Overall health assessment
        system_health["overall_healthy"] = all(
            [health_status["healthy"], recent_predictions > 0, reference_data_count > 0]
        )

        task_logger.info(
            f"System health: {'HEALTHY' if system_health['overall_healthy'] else 'ISSUES DETECTED'}"
        )
        return system_health

    except Exception as e:
        task_logger.error(f"Health check failed: {str(e)}")
        return {
            "overall_healthy": False,
            "check_failed": True,
            "error": str(e),
            "check_timestamp": datetime.now().isoformat(),
        }


@task
def generate_monitoring_report_task(
    drift_analysis: Dict[str, Any], health_status: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate simple monitoring report.

    Args:
        drift_analysis: Results from drift trend analysis.
        health_status: Results from health check.

    Returns:
        Dict[str, Any]: Monitoring report.
    """
    task_logger = get_run_logger()
    task_logger.info("Generating monitoring report")

    # Generate text report
    report_lines = [
        "=" * 60,
        "SOLAR FORECASTING MONITORING REPORT",
        "=" * 60,
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SYSTEM HEALTH:",
        f"  Overall Status: {'✅ HEALTHY' if health_status.get('overall_healthy', False) else '❌ ISSUES'}",
        f"  Database: {'✅ OK' if health_status.get('database_healthy', False) else '❌ ERROR'}",
        f"  Reference Data: {health_status.get('reference_data_count', 0)} records",
        f"  Recent Predictions: {health_status.get('recent_predictions_24h', 0)} (24h)",
        "",
    ]

    if not drift_analysis.get("analysis_failed", False):
        report_lines.extend(
            [
                "DRIFT ANALYSIS:",
                f"  Period: {drift_analysis['analysis_period_days']} days",
                f"  Overall Drift Rate: {drift_analysis['overall_drift_rate']}%",
                f"  Total Drift Events: {drift_analysis['total_drift_events']}",
                "",
                "TOP DRIFTED FEATURES:",
            ]
        )

        for feature in drift_analysis.get("top_drifted_features", [])[:3]:
            report_lines.append(
                f"  - {feature['feature_name']}: {feature['drift_count']} events"
            )
    else:
        report_lines.extend(
            [
                "DRIFT ANALYSIS:",
                "  ❌ Analysis failed - check system logs",
            ]
        )

    report_lines.extend(
        [
            "",
            "RECOMMENDATIONS:",
        ]
    )

    # Simple recommendations based on data
    if drift_analysis.get("overall_drift_rate", 0) > 20:
        report_lines.append("  ⚠️  High drift rate detected - consider model retraining")

    if health_status.get("recent_predictions_24h", 0) == 0:
        report_lines.append(
            "  ⚠️  No recent predictions - check batch prediction system"
        )

    if not health_status.get("reference_data_available", False):
        report_lines.append("  ⚠️  No reference data - run model training")

    if drift_analysis.get("overall_drift_rate", 0) < 10 and health_status.get(
        "overall_healthy", False
    ):
        report_lines.append("  ✅ System operating normally")

    report_lines.append("=" * 60)

    # Log the complete report
    for line in report_lines:
        task_logger.info(line)

    monitoring_report = {
        "report_timestamp": datetime.now().isoformat(),
        "drift_analysis": drift_analysis,
        "health_status": health_status,
        "report_text": "\n".join(report_lines),
        "summary": {
            "overall_healthy": health_status.get("overall_healthy", False),
            "drift_rate": drift_analysis.get("overall_drift_rate", 0),
            "recommendations_count": len(
                [
                    line
                    for line in report_lines
                    if line.startswith("  ⚠️") or line.startswith("  ✅")
                ]
            ),
        },
    }

    return monitoring_report


@flow(
    name="solar-monitoring-flow",
    description="Basic monitoring flow for solar forecasting system",
    version="1.0",
)
def solar_monitoring_flow(analysis_days: int = 7) -> Dict[str, Any]:
    """
    Basic monitoring flow for solar forecasting system.

    Args:
        analysis_days: Number of days to analyze for drift trends.

    Returns:
        Dict[str, Any]: Complete monitoring results.
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"Starting solar monitoring flow (analyzing {analysis_days} days)")

    try:
        # Task 1: Analyze drift trends
        drift_analysis = analyze_drift_trends_task(analysis_days)

        # Task 2: System health check
        health_status = system_health_check_task()

        # Task 3: Generate monitoring report
        monitoring_report = generate_monitoring_report_task(
            drift_analysis, health_status
        )

        flow_logger.info("Solar monitoring flow completed successfully")
        return monitoring_report

    except Exception as e:
        flow_logger.error(f"Solar monitoring flow failed: {str(e)}")
        raise


def main() -> None:
    """
    CLI entry point for monitoring flow.

    Example usage:
        python -m src.monitoring.monitor_flow
        python -m src.monitoring.monitor_flow 14  # Analyze 14 days
    """
    import sys

    # Parse command line arguments
    analysis_days = 7
    if len(sys.argv) > 1:
        try:
            analysis_days = int(sys.argv[1])
        except ValueError:
            print("Usage: python -m src.monitoring.monitor_flow [days]")
            sys.exit(1)

    print(f"Running solar monitoring flow (analyzing {analysis_days} days)...")

    # Run the monitoring flow
    result = solar_monitoring_flow(analysis_days)

    print("\nMonitoring flow completed!")
    if result.get("summary"):
        summary = result["summary"]
        print(
            f"System Health: {'✅ HEALTHY' if summary['overall_healthy'] else '❌ ISSUES'}"
        )
        print(f"Drift Rate: {summary['drift_rate']}%")
        print(f"Recommendations: {summary['recommendations_count']}")


if __name__ == "__main__":
    main()


# Export main flow
__all__ = ["solar_monitoring_flow"]
