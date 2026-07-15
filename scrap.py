
    def _create_sentinel_plan(self, train_dataset: dict[str, Dataset] | None) -> SentinelPlan:
        """
        Generates and returns a SentinelPlan for the given dataset.
        """
        # create a new optimizer and update its strategy to SENTINEL
        optimizer = self.optimizer.deepcopy_clean()
        optimizer.update_strategy(OptimizationStrategyType.SENTINEL)

        # create copy of dataset, but change its root Dataset(s) to the validation Dataset(s)
        dataset = self.dataset.copy()
        if train_dataset is not None:
            dataset._set_root_datasets(train_dataset)
            dataset._generate_unique_logical_op_ids()

        # get the sentinel plan for the given dataset
        sentinel_plans = optimizer.optimize(dataset)
        sentinel_plan = sentinel_plans[0]

        return sentinel_plan

    def execute(self) -> DataRecordCollection:
        logger.info(f"Executing {self.__class__.__name__}")

        # create execution stats
        execution_stats = ExecutionStats(execution_id=self.execution_id())
        execution_stats.start()

        # if the user provides a validator, we perform optimization
        if self.validator is not None:
            # create sentinel plan
            sentinel_plan = self._create_sentinel_plan(self.train_dataset)

            # generate sample execution data
            if self.train_dataset is not None:
                sentinel_plan_stats = self.sentinel_execution_strategy.execute_sentinel_plan(sentinel_plan, self.train_dataset, self.validator)

            else:
                train_dataset = self.dataset._get_root_datasets()
                sentinel_plan_stats = self.sentinel_execution_strategy.execute_sentinel_plan(sentinel_plan, train_dataset, self.validator)

            # update the execution stats to account for the work done in optimization
            execution_stats.add_plan_stats(sentinel_plan_stats)
            execution_stats.finish_optimization()

            # (re-)initialize the optimizer
            self.optimizer = self.optimizer.deepcopy_clean()

            # construct the CostModel with any sample execution data we've gathered
            cost_model = SampleBasedCostModel(sentinel_plan_stats, self.verbose)
            self.optimizer.cost_model = cost_model 
            # update_cost_model(cost_model)

        # get the optimal plan according to the optimizer
        plans = self.optimizer.optimize(self.dataset)
        final_plan = plans[0]
        records, plan_stats = self.execution_strategy.execute_plan(plan=final_plan)

        # update the execution stats to account for the work to execute the final plan
        execution_stats.add_plan_stats([plan_stats])
        execution_stats.finish()

        # construct and return the DataRecordCollection
        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info(f"Done executing {self.__class__.__name__}")

        return result