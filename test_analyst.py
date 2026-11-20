import unittest

from analyst import (
    analyze,
    load_kaggle_data,
    make_demo_data,
    prepare_dataset,
)


class AnalystGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_demo_data()

    def test_ranks_products_for_latest_quarter(self):
        result = analyze("What were the top 3 products by revenue last quarter?", self.data)

        self.assertEqual(result["plan"], ["rank"])
        self.assertEqual(result["metric"], "Sales")
        self.assertEqual(result["top_n"], 3)
        self.assertEqual(result["results"][0]["tool"], "rank")
        self.assertEqual(len(result["results"][0]["data"]["products"]), 3)
        self.assertIn("latest available quarter, Q4 2017", result["answer"])

    def test_filters_a_category_and_computes_seasonality(self):
        result = analyze(
            "Is there a seasonal trend in sales for category Technology?", self.data
        )

        self.assertEqual(result["plan"], ["trend"])
        self.assertEqual(result["filters"], {"Category": ["Technology"]})
        self.assertIn("month-of-year averages", result["answer"])
        self.assertIn("Technology", result["plan_text"])

    def test_compares_regions_and_explains_performance_drivers(self):
        result = analyze(
            "Compare East and West performance and tell me what's driving the difference",
            self.data,
        )

        self.assertEqual(result["plan"], ["compare", "drivers"])
        self.assertEqual(result["focus_values"], ["East", "West"])
        self.assertIn("largest Sub-Category differences", result["answer"])

    def test_selects_anomaly_and_overview_for_a_multi_step_question(self):
        result = analyze("Find unusual values and summarize the data", self.data)

        self.assertEqual(result["plan"], ["anomaly", "overview"])
        self.assertIn("rows) using the IQR rule", result["answer"])
        self.assertIn(f"{len(self.data):,} rows", result["answer"])

    def test_missing_date_is_reported_instead_of_failing(self):
        data = self.data.drop(columns=["Order Date"])
        result = analyze("Show the Sales trend over time", data)

        self.assertEqual(result["plan"], ["trend"])
        self.assertIn("needs a date or time column", result["answer"])


class SuppliedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_kaggle_data()
        cls.prepared, cls.date_note = prepare_dataset(cls.data)

    def test_loads_supplied_global_superstore_csv(self):
        self.assertEqual(len(self.data), 51290)
        self.assertIn("Sales", self.data.columns)
        self.assertIn("Region", self.data.columns)
        self.assertIn("Product Name", self.data.columns)

    def test_reconstructs_week_proxy_and_discloses_date_limit(self):
        self.assertIn("Analysis Week", self.prepared.columns)
        self.assertIn("approximate week midpoints", self.date_note)
        self.assertEqual(self.prepared["Analysis Week"].min().year, 2011)
        self.assertEqual(self.prepared["Analysis Week"].max().year, 2014)

    def test_ranks_actual_products_in_latest_available_quarter(self):
        result = analyze(
            "What were the top 3 products by sales last quarter?", self.data
        )

        self.assertEqual(result["plan"], ["rank"])
        self.assertEqual(result["results"][0]["data"]["period"], "latest available quarter, Q4 2014")
        self.assertEqual(len(result["results"][0]["data"]["products"]), 3)
        self.assertIn("Time-data caveat", result["answer"])

    def test_infers_region_from_named_values_and_explains_drivers(self):
        result = analyze(
            "Compare East and West performance and tell me what's driving the difference",
            self.data,
        )

        self.assertEqual(result["group"], "Region")
        self.assertEqual(result["focus_values"], ["East", "West"])
        self.assertIn("largest Sub-Category differences", result["answer"])

    def test_seasonality_uses_only_the_named_category(self):
        result = analyze(
            "Is there a seasonal trend in sales for category Technology?", self.data
        )

        self.assertEqual(result["filters"], {"Category": ["Technology"]})
        self.assertIn("month-of-year averages", result["answer"])
        self.assertIn("Time-data caveat", result["answer"])


if __name__ == "__main__":
    unittest.main()
