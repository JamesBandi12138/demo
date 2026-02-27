import os
import pandas as pd


class DataCleaner:

    def __init__(self):
        self.df = None
        self.original_df = None
        self.file_path = None

    def load_file(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            print(f"文件不存在 - {file_path}")
            return False

        self.file_path = file_path
        ext = Path(file_path).suffix.lower()

        try:
            if ext == '.csv':
                self.df = pd.read_csv(file_path, encoding='utf-8')
            elif ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path, engine='openpyxl')
            else:
                print(f"不支持该格式 - {ext}")
                return False

            self.original_df = self.df.copy()
            print(f"加载完成: {file_path}")
            print(f"共 {self.df.shape[0]} 行, {self.df.shape[1]} 列")
            return True

        except Exception as e:
            print(f"读取失败: {e}")
            return False

    def show_info(self):
        if self.df is None:
            print("请先加载数据")
            return

        print("\n" + "="*50)
        print("数据概览")
        print("="*50)
        print(f"行数: {len(self.df)}")
        print(f"列数: {len(self.df.columns)}")
        print(f"列名: {list(self.df.columns)}")
        print(f"\n数据类型:\n{self.df.dtypes}")
        print(f"\n缺失值:\n{self.df.isnull().sum()}")
        print(f"\n重复行: {self.df.duplicated().sum()}")
        print("="*50)

    def handle_missing_values(self, strategy: str = 'drop', fill_value=None, columns=None):
        if self.df is None:
            print("请先加载数据")
            return False

        target_cols = columns if columns else self.df.columns.tolist()
        before_count = self.df.isnull().sum().sum()

        try:
            if strategy == 'drop':
                self.df.dropna(subset=target_cols, inplace=True)
                print("已删除含缺失值的行")

            elif strategy == 'fill' and fill_value is not None:
                self.df[target_cols] = self.df[target_cols].fillna(fill_value)
                print(f"已用 '{fill_value}' 填充")

            elif strategy == 'mean':
                for col in target_cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                print("已用均值填充数值列")

            elif strategy == 'median':
                for col in target_cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                print("已用中位数填充数值列")

            elif strategy == 'mode':
                for col in target_cols:
                    mode_val = self.df[col].mode()
                    if not mode_val.empty:
                        self.df[col].fillna(mode_val[0], inplace=True)
                print("已用众数填充")

            after_count = self.df.isnull().sum().sum()
            print(f"缺失值: {before_count} -> {after_count}")
            return True

        except Exception as e:
            print(f"处理失败: {e}")
            return False

    def remove_duplicates(self, subset=None, keep: str = 'first'):
        if self.df is None:
            print("请先加载数据")
            return False

        before_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        after_count = len(self.df)

        print(f"删除重复行: {before_count - after_count} 行")
        print(f"剩余: {after_count} 行")
        return True

    def convert_types(self, column: str, dtype: str):
        if self.df is None:
            print("请先加载数据")
            return False

        if column not in self.df.columns:
            print(f"列不存在 - {column}")
            return False

        try:
            if dtype == 'int':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce').astype('Int64')
            elif dtype == 'float':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            elif dtype == 'str':
                self.df[column] = self.df[column].astype(str)
            elif dtype == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
            else:
                print(f"不支持该类型 - {dtype}")
                return False

            print(f"已将 '{column}' 转为 {dtype}")
            return True

        except Exception as e:
            print(f"转换失败: {e}")
            return False

    def rename_columns(self, rename_map: dict):
        if self.df is None:
            print("请先加载数据")
            return False

        self.df.rename(columns=rename_map, inplace=True)
        print(f"已重命名: {rename_map}")
        return True

    def select_columns(self, columns: list):
        if self.df is None:
            print("请先加载数据")
            return False

        invalid_cols = [col for col in columns if col not in self.df.columns]
        if invalid_cols:
            print(f"这些列不存在 - {invalid_cols}")

        valid_cols = [col for col in columns if col in self.df.columns]
        self.df = self.df[valid_cols]
        print(f"已选择: {valid_cols}")
        return True

    def filter_rows(self, column: str, operator: str, value):
        if self.df is None:
            print("请先加载数据")
            return False

        if column not in self.df.columns:
            print(f"列不存在 - {column}")
            return False

        before_count = len(self.df)

        try:
            if operator == '==':
                self.df = self.df[self.df[column] == value]
            elif operator == '!=':
                self.df = self.df[self.df[column] != value]
            elif operator == '>':
                self.df = self.df[self.df[column] > value]
            elif operator == '<':
                self.df = self.df[self.df[column] < value]
            elif operator == '>=':
                self.df = self.df[self.df[column] >= value]
            elif operator == '<=':
                self.df = self.df[self.df[column] <= value]
            elif operator == 'contains':
                self.df = self.df[self.df[column].astype(str).str.contains(str(value), na=False)]
            else:
                print(f"不支持该操作符 - {operator}")
                return False

            after_count = len(self.df)
            print(f"筛选完成: {before_count} -> {after_count} 行")
            return True

        except Exception as e:
            print(f"筛选失败: {e}")
            return False

    def reset_data(self):
        if self.original_df is None:
            print("没有原始数据")
            return False

        self.df = self.original_df.copy()
        print("已重置")
        return True

    def export_file(self, output_path: str, index: bool = False):
        if self.df is None:
            print("请先加载数据")
            return False

        ext = Path(output_path).suffix.lower()

        try:
            if ext == '.csv':
                self.df.to_csv(output_path, index=index, encoding='utf-8-sig')
            elif ext in ['.xlsx', '.xls']:
                self.df.to_excel(output_path, index=index, engine='openpyxl')
            else:
                print(f"不支持该格式 - {ext}")
                return False

            print(f"已导出: {output_path}")
            return True

        except Exception as e:
            print(f"导出失败: {e}")
            return False


def print_menu():
    print("\n" + "="*50)
    print("数据清洗工具")
    print("="*50)
    print("1. 加载文件")
    print("2. 查看数据")
    print("3. 处理缺失值")
    print("4. 删除重复行")
    print("5. 转换类型")
    print("6. 重命名列")
    print("7. 选择列")
    print("8. 筛选行")
    print("9. 重置数据")
    print("10. 导出文件")
    print("0. 退出")
    print("="*50)


def get_input(prompt: str, default=None):
    if default:
        result = input(f"{prompt} (默认: {default}): ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def main():
    cleaner = DataCleaner()

    while True:
        print_menu()
        choice = get_input("选择操作")

        if choice == '0':
            print("再见!")
            break

        elif choice == '1':
            file_path = get_input("文件路径")
            if file_path:
                cleaner.load_file(file_path)

        elif choice == '2':
            cleaner.show_info()

        elif choice == '3':
            print("\n缺失值处理:")
            print("1. 删除")
            print("2. 填充指定值")
            print("3. 均值填充")
            print("4. 中位数填充")
            print("5. 众数填充")

            strategy_choice = get_input("选择策略")
            strategy_map = {'1': 'drop', '2': 'fill', '3': 'mean', '4': 'median', '5': 'mode'}

            if strategy_choice in strategy_map:
                strategy = strategy_map[strategy_choice]
                fill_value = None

                if strategy == 'fill':
                    fill_value = get_input("填充值")

                columns_input = get_input("指定列 (逗号分隔，留空为全部)")
                columns = [c.strip() for c in columns_input.split(',')] if columns_input else None

                cleaner.handle_missing_values(strategy, fill_value, columns)

        elif choice == '4':
            subset_input = get_input("判断重复的列 (逗号分隔，留空为全部)")
            subset = [c.strip() for c in subset_input.split(',')] if subset_input else None

            print("保留: 1.第一个  2.最后一个")
            keep_choice = get_input("选择", '1')
            keep = 'first' if keep_choice == '1' else 'last'

            cleaner.remove_duplicates(subset, keep)

        elif choice == '5':
            if cleaner.df is not None:
                print(f"可用列: {list(cleaner.df.columns)}")
            column = get_input("列名")
            print("类型: 1.int  2.float  3.str  4.datetime")
            type_choice = get_input("选择类型")
            type_map = {'1': 'int', '2': 'float', '3': 'str', '4': 'datetime'}

            if column and type_choice in type_map:
                cleaner.convert_types(column, type_map[type_choice])

        elif choice == '6':
            if cleaner.df is not None:
                print(f"当前列名: {list(cleaner.df.columns)}")
            old_name = get_input("原列名")
            new_name = get_input("新列名")

            if old_name and new_name:
                cleaner.rename_columns({old_name: new_name})

        elif choice == '7':
            if cleaner.df is not None:
                print(f"可用列: {list(cleaner.df.columns)}")
            columns_input = get_input("保留的列 (逗号分隔)")
            columns = [c.strip() for c in columns_input.split(',') if c.strip()]

            if columns:
                cleaner.select_columns(columns)

        elif choice == '8':
            if cleaner.df is not None:
                print(f"可用列: {list(cleaner.df.columns)}")
            column = get_input("列名")
            print("操作: 1.==  2.!=  3.>  4.<  5.>=  6.<=  7.包含")
            op_choice = get_input("选择操作")
            op_map = {'1': '==', '2': '!=', '3': '>', '4': '<', '5': '>=', '6': '<=', '7': 'contains'}
            value = get_input("比较值")

            if column and op_choice in op_map and value:
                cleaner.filter_rows(column, op_map[op_choice], value)

        elif choice == '9':
            cleaner.reset_data()

        elif choice == '10':
            output_path = get_input("输出路径 (如: output.csv)")
            if output_path:
                cleaner.export_file(output_path)

        else:
            print("无效选择")


if __name__ == '__main__':
    main()
