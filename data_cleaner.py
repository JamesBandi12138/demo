#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据清洗工具 - 简洁实用的数据处理工具
支持CSV和Excel文件的读取、清洗和导出
"""

import os
import sys
from pathlib import Path

import pandas as pd


class DataCleaner:
    """数据清洗核心类"""

    def __init__(self):
        self.df = None
        self.original_df = None
        self.file_path = None

    def load_file(self, file_path: str) -> bool:
        """根据文件扩展名自动读取CSV或Excel文件"""
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return False

        self.file_path = file_path
        ext = Path(file_path).suffix.lower()

        try:
            if ext == '.csv':
                self.df = pd.read_csv(file_path, encoding='utf-8')
            elif ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path, engine='openpyxl')
            else:
                print(f"错误: 不支持的文件格式 - {ext}")
                return False

            self.original_df = self.df.copy()
            print(f"成功加载文件: {file_path}")
            print(f"数据维度: {self.df.shape[0]} 行 x {self.df.shape[1]} 列")
            return True

        except Exception as e:
            print(f"读取文件失败: {e}")
            return False

    def show_info(self):
        """显示数据基本信息"""
        if self.df is None:
            print("请先加载数据文件")
            return

        print("\n" + "="*50)
        print("数据基本信息")
        print("="*50)
        print(f"行数: {len(self.df)}")
        print(f"列数: {len(self.df.columns)}")
        print(f"\n列名: {list(self.df.columns)}")
        print(f"\n数据类型:\n{self.df.dtypes}")
        print(f"\n缺失值统计:\n{self.df.isnull().sum()}")
        print(f"\n重复行数: {self.df.duplicated().sum()}")
        print("="*50)

    def handle_missing_values(self, strategy: str = 'drop', fill_value=None, columns=None):
        """
        处理缺失值
        
        Args:
            strategy: 处理策略 - 'drop'(删除), 'fill'(填充), 'mean'(均值), 'median'(中位数), 'mode'(众数)
            fill_value: 当strategy='fill'时使用的填充值
            columns: 指定处理的列，None表示所有列
        """
        if self.df is None:
            print("请先加载数据文件")
            return False

        target_cols = columns if columns else self.df.columns.tolist()
        before_count = self.df.isnull().sum().sum()

        try:
            if strategy == 'drop':
                self.df.dropna(subset=target_cols, inplace=True)
                print(f"已删除包含缺失值的行")

            elif strategy == 'fill' and fill_value is not None:
                self.df[target_cols] = self.df[target_cols].fillna(fill_value)
                print(f"已用 '{fill_value}' 填充缺失值")

            elif strategy == 'mean':
                for col in target_cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                print("已用均值填充数值列的缺失值")

            elif strategy == 'median':
                for col in target_cols:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                print("已用中位数填充数值列的缺失值")

            elif strategy == 'mode':
                for col in target_cols:
                    mode_val = self.df[col].mode()
                    if not mode_val.empty:
                        self.df[col].fillna(mode_val[0], inplace=True)
                print("已用众数填充缺失值")

            after_count = self.df.isnull().sum().sum()
            print(f"缺失值数量: {before_count} -> {after_count}")
            return True

        except Exception as e:
            print(f"处理缺失值失败: {e}")
            return False

    def remove_duplicates(self, subset=None, keep: str = 'first'):
        """
        删除重复行
        
        Args:
            subset: 指定判断重复的列，None表示所有列
            keep: 保留策略 - 'first'(保留第一个), 'last'(保留最后一个), False(删除所有重复)
        """
        if self.df is None:
            print("请先加载数据文件")
            return False

        before_count = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        after_count = len(self.df)

        print(f"已删除重复行: {before_count - after_count} 行")
        print(f"当前数据: {after_count} 行")
        return True

    def convert_types(self, column: str, dtype: str):
        """
        转换数据类型
        
        Args:
            column: 列名
            dtype: 目标类型 - 'int', 'float', 'str', 'datetime'
        """
        if self.df is None:
            print("请先加载数据文件")
            return False

        if column not in self.df.columns:
            print(f"错误: 列 '{column}' 不存在")
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
                print(f"错误: 不支持的数据类型 - {dtype}")
                return False

            print(f"已将列 '{column}' 转换为 {dtype} 类型")
            return True

        except Exception as e:
            print(f"类型转换失败: {e}")
            return False

    def rename_columns(self, rename_map: dict):
        """重命名列"""
        if self.df is None:
            print("请先加载数据文件")
            return False

        self.df.rename(columns=rename_map, inplace=True)
        print(f"已重命名列: {rename_map}")
        return True

    def select_columns(self, columns: list):
        """选择保留的列"""
        if self.df is None:
            print("请先加载数据文件")
            return False

        invalid_cols = [col for col in columns if col not in self.df.columns]
        if invalid_cols:
            print(f"警告: 以下列不存在 - {invalid_cols}")

        valid_cols = [col for col in columns if col in self.df.columns]
        self.df = self.df[valid_cols]
        print(f"已选择列: {valid_cols}")
        return True

    def filter_rows(self, column: str, operator: str, value):
        """
        筛选行
        
        Args:
            column: 列名
            operator: 操作符 - '==', '!=', '>', '<', '>=', '<=', 'contains'
            value: 比较值
        """
        if self.df is None:
            print("请先加载数据文件")
            return False

        if column not in self.df.columns:
            print(f"错误: 列 '{column}' 不存在")
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
                print(f"错误: 不支持的操作符 - {operator}")
                return False

            after_count = len(self.df)
            print(f"筛选完成: {before_count} -> {after_count} 行")
            return True

        except Exception as e:
            print(f"筛选失败: {e}")
            return False

    def reset_data(self):
        """重置为原始数据"""
        if self.original_df is None:
            print("没有原始数据可重置")
            return False

        self.df = self.original_df.copy()
        print("已重置为原始数据")
        return True

    def export_file(self, output_path: str, index: bool = False):
        """导出数据到CSV或Excel文件"""
        if self.df is None:
            print("请先加载数据文件")
            return False

        ext = Path(output_path).suffix.lower()

        try:
            if ext == '.csv':
                self.df.to_csv(output_path, index=index, encoding='utf-8-sig')
            elif ext in ['.xlsx', '.xls']:
                self.df.to_excel(output_path, index=index, engine='openpyxl')
            else:
                print(f"错误: 不支持的输出格式 - {ext}")
                return False

            print(f"数据已导出至: {output_path}")
            return True

        except Exception as e:
            print(f"导出失败: {e}")
            return False


def print_menu():
    """打印主菜单"""
    print("\n" + "="*50)
    print("数据清洗工具")
    print("="*50)
    print("1. 加载文件 (CSV/Excel)")
    print("2. 查看数据信息")
    print("3. 处理缺失值")
    print("4. 删除重复行")
    print("5. 转换数据类型")
    print("6. 重命名列")
    print("7. 选择列")
    print("8. 筛选行")
    print("9. 重置数据")
    print("10. 导出文件")
    print("0. 退出")
    print("="*50)


def get_input(prompt: str, default=None):
    """获取用户输入，支持默认值"""
    if default:
        result = input(f"{prompt} (默认: {default}): ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def main():
    """主程序入口"""
    cleaner = DataCleaner()

    while True:
        print_menu()
        choice = get_input("请选择操作")

        if choice == '0':
            print("感谢使用，再见!")
            break

        elif choice == '1':
            file_path = get_input("请输入文件路径")
            if file_path:
                cleaner.load_file(file_path)

        elif choice == '2':
            cleaner.show_info()

        elif choice == '3':
            print("\n缺失值处理策略:")
            print("1. 删除包含缺失值的行")
            print("2. 用指定值填充")
            print("3. 用均值填充 (数值列)")
            print("4. 用中位数填充 (数值列)")
            print("5. 用众数填充")

            strategy_choice = get_input("请选择策略")
            strategy_map = {'1': 'drop', '2': 'fill', '3': 'mean', '4': 'median', '5': 'mode'}

            if strategy_choice in strategy_map:
                strategy = strategy_map[strategy_choice]
                fill_value = None

                if strategy == 'fill':
                    fill_value = get_input("请输入填充值")

                columns_input = get_input("指定列 (多个列用逗号分隔，留空表示所有列)")
                columns = [c.strip() for c in columns_input.split(',')] if columns_input else None

                cleaner.handle_missing_values(strategy, fill_value, columns)

        elif choice == '4':
            subset_input = get_input("判断重复的列 (多个列用逗号分隔，留空表示所有列)")
            subset = [c.strip() for c in subset_input.split(',')] if subset_input else None

            print("保留策略: 1.第一个 2.最后一个")
            keep_choice = get_input("请选择", '1')
            keep = 'first' if keep_choice == '1' else 'last'

            cleaner.remove_duplicates(subset, keep)

        elif choice == '5':
            if cleaner.df is not None:
                print(f"可用列: {list(cleaner.df.columns)}")
            column = get_input("请输入列名")
            print("数据类型: 1.int  2.float  3.str  4.datetime")
            type_choice = get_input("请选择类型")
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
            columns_input = get_input("请输入要保留的列 (用逗号分隔)")
            columns = [c.strip() for c in columns_input.split(',') if c.strip()]

            if columns:
                cleaner.select_columns(columns)

        elif choice == '8':
            if cleaner.df is not None:
                print(f"可用列: {list(cleaner.df.columns)}")
            column = get_input("请输入列名")
            print("操作符: 1.==  2.!=  3.>  4.<  5.>=  6.<=  7.contains")
            op_choice = get_input("请选择操作符")
            op_map = {'1': '==', '2': '!=', '3': '>', '4': '<', '5': '>=', '6': '<=', '7': 'contains'}
            value = get_input("请输入比较值")

            if column and op_choice in op_map and value:
                cleaner.filter_rows(column, op_map[op_choice], value)

        elif choice == '9':
            cleaner.reset_data()

        elif choice == '10':
            output_path = get_input("请输入输出文件路径 (如: output.csv 或 output.xlsx)")
            if output_path:
                cleaner.export_file(output_path)

        else:
            print("无效选择，请重新输入")


if __name__ == '__main__':
    main()
