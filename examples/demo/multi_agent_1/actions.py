# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemoguardrails.actions import action


@action(name="SearchAction")
async def action(query: str):
    return [
        {
            "url": "https://www.statista.com/statistics/281744/gdp-of-the-united-kingdom",
            "content": "The gross domestic product of the United Kingdom in 2023 was 2.27 trillion British pounds, a slight increase when compared to the previous year.",
        }
    ]
