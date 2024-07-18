from palimpzest.planner import LogicalPlanner
from palimpzest.operators import BaseScan, ConvertScan, FilteredScan
import palimpzest as pz
import pytest

class TestLogicalPlanner:

    def test_basic_functionality(self, enron_eval_tiny, email_schema):
        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 1

        plan = logical_plans[0]
        assert len(plan) == 2
        assert isinstance(plan[0], BaseScan)
        assert isinstance(plan[1], ConvertScan)

    def test_static_convert_order(self, enron_eval_tiny, email_schema):
        class SomeSchema(pz.Schema):
            new_field = pz.StringField("some field")

        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.convert(SomeSchema)
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 1

        plan = logical_plans[0]
        assert len(plan) == 3
        assert isinstance(plan[0], BaseScan)
        assert isinstance(plan[1], ConvertScan)
        assert plan[1].outputSchema == email_schema
        assert isinstance(plan[2], ConvertScan)
        assert plan[2].outputSchema == SomeSchema

    def test_basic_mixed_operator(self, enron_eval_tiny, email_schema):
        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.filter("some filter")
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 1

        plan = logical_plans[0]
        assert len(plan) == 3
        assert isinstance(plan[0], BaseScan)
        assert isinstance(plan[1], ConvertScan)
        assert isinstance(plan[2], FilteredScan)

    def test_downtream_op_presence(self, enron_eval_tiny, email_schema):
        class SomeSchema(pz.Schema):
            new_field = pz.StringField("some field")

        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.filter("some filter")
        dataset = dataset.convert(SomeSchema)
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 1

        plan = logical_plans[0]
        assert len(plan) == 4
        assert isinstance(plan[0], BaseScan)
        assert isinstance(plan[1], ConvertScan)
        assert plan[1].outputSchema == email_schema
        assert isinstance(plan[2], FilteredScan)
        assert isinstance(plan[3], ConvertScan)
        assert plan[3].outputSchema == SomeSchema

    def test_basic_filter_reordering(self, enron_eval_tiny, email_schema):
        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.filter("some filter")
        dataset = dataset.filter("another filter")
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 2
        for plan in logical_plans:
            assert len(plan) == 4
            assert isinstance(plan[0], BaseScan)
            assert isinstance(plan[1], ConvertScan)
            assert isinstance(plan[2], FilteredScan)
            assert isinstance(plan[3], FilteredScan)
        
        first_plan, second_plan = logical_plans[0], logical_plans[1]
        assert first_plan[2].filter == second_plan[3].filter
        assert first_plan[3].filter == second_plan[2].filter

    def test_multi_parent_simple(self, enron_eval_tiny, email_schema):
        class SomeSchema(pz.Schema):
            new_field = pz.StringField("some field")

        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.convert(SomeSchema)
        dataset = dataset.filter("some filter", depends_on=["sender", "new_field"])
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 1

        plan = logical_plans[0]
        assert len(plan) == 4
        assert isinstance(plan[0], BaseScan)
        assert isinstance(plan[1], ConvertScan)
        assert isinstance(plan[2], ConvertScan)
        assert isinstance(plan[3], FilteredScan)

    def test_multi_child_simple(self, enron_eval_tiny, email_schema):
        class SomeSchema(pz.Schema):
            new_field = pz.StringField("some field")

        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.convert(SomeSchema, depends_on=["contents"])
        dataset = dataset.filter("some filter", depends_on=["contents"])
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 1

        plan = logical_plans[0]
        assert len(plan) == 4
        assert isinstance(plan[0], BaseScan)
        assert isinstance(plan[1], FilteredScan)
        assert isinstance(plan[2], ConvertScan)
        assert isinstance(plan[3], ConvertScan)

    def test_complex1(self, enron_eval_tiny, email_schema):
        """
        Test Plan dependency graph:
            S --> C1 --> F1
              \
               --> C2 --> F2

        Expected logical plans:
        - S --> C1 --> F1 --> C2 --> F2
        - S --> C2 --> F2 --> C1 --> F1
        """
        class SomeSchema(pz.Schema):
            new_field = pz.StringField("some field")

        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.convert(SomeSchema, depends_on=["contents"])
        dataset = dataset.filter("some filter1", depends_on=["sender"])
        dataset = dataset.filter("some filter2", depends_on=["new_field"])
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 2
        for plan in logical_plans:
            assert len(plan) == 5
            assert isinstance(plan[0], BaseScan)
            assert isinstance(plan[1], ConvertScan)
            assert isinstance(plan[2], FilteredScan)
            assert isinstance(plan[3], ConvertScan)
            assert isinstance(plan[4], FilteredScan)

        first_plan, second_plan = logical_plans[0], logical_plans[1]
        assert first_plan[2].filter == second_plan[4].filter
        assert first_plan[4].filter == second_plan[2].filter

    def test_complex2(self, enron_eval_tiny, email_schema):
        """
        Test Plan dependency graph:
            S --> C1 --> F1
              \
               --> C2 --> F2
               \
                --> F3 

        Expected logical plans:
        - S --> C1 --> F1 --> C2 --> F2 --> F3
        - S --> C1 --> F1 --> F3 --> C2 --> F2
        - S --> C2 --> F2 --> C1 --> F1 --> F3
        - S --> C2 --> F2 --> F3 --> C1 --> F1
        - S --> F3 --> C1 --> F1 --> C2 --> F2
        - S --> F3 --> C2 --> F2 --> C1 --> F1
        """
        class SomeSchema(pz.Schema):
            new_field = pz.StringField("some field")

        dataset = pz.Dataset(enron_eval_tiny, schema=email_schema)
        dataset = dataset.convert(SomeSchema, depends_on=["contents"])
        dataset = dataset.filter("some filter1", depends_on=["sender"])
        dataset = dataset.filter("some filter2", depends_on=["new_field"])
        dataset = dataset.filter("some filter3", depends_on=["contents"])
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 6
        for plan in logical_plans:
            assert len(plan) == 6

    def test_complex3(self, enron_eval_tiny):
        """
        Test Plan dependency graph:
            S --> C1 --> C2 --> F1 --> C3 --> C4 --> F2
              \\      /---------/
               --> C5 --> F3
               \
                --> F4
                \
                 --> C6 --> C7

        Expected logical plans: 4!
        NOTE: for now, we allow the plan which executes C1 --> C2 --> C3 --> C4 --> F2 --> F1;
              the effort required to disallow this plan is more than trivial, and I plan on
              overhauling the planning logic in the near future to become an optimization problem anyways.
        """
        class SomeSchema1(pz.Schema):
            new_field1 = pz.StringField("some field1")
        
        class SomeSchema2(pz.Schema):
            new_field2 = pz.StringField("some field2")

        class SomeSchema3(pz.Schema):
            new_field3 = pz.StringField("some field3")

        class SomeSchema4(pz.Schema):
            new_field4 = pz.StringField("some field4")

        class SomeSchema5(pz.Schema):
            new_field5 = pz.StringField("some field5")

        class SomeSchema6(pz.Schema):
            new_field6 = pz.StringField("some field6")

        class SomeSchema7(pz.Schema):
            new_field7 = pz.StringField("some field7")

        dataset = pz.Dataset(enron_eval_tiny, schema=SomeSchema1)
        dataset = dataset.convert(SomeSchema2)
        dataset = dataset.convert(SomeSchema5, depends_on=['contents'])
        dataset = dataset.filter("some filter1", depends_on=['new_field2', 'new_field5'])
        dataset = dataset.convert(SomeSchema3)
        dataset = dataset.convert(SomeSchema4)
        dataset = dataset.filter("some filter2")
        dataset = dataset.filter("some filter3", depends_on=['new_field5'])
        dataset = dataset.filter("some filter4", depends_on=['contents'])
        dataset = dataset.convert(SomeSchema6, depends_on=['contents'])
        dataset = dataset.convert(SomeSchema7, depends_on=['new_field6'])
        logical_planner = LogicalPlanner(no_cache=True, verbose=True)
        logical_plans = logical_planner.generate_plans(dataset)

        assert len(logical_plans) == 24
        for plan in logical_plans:
            assert len(plan) == 12
