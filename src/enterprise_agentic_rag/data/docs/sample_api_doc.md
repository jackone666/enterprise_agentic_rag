# 企业 API 网关文档

## 认证方式
### Bearer Token
Authorization: Bearer <your_token>

### API Key
X-API-Key: <your_api_key>

## 核心接口
### GET /api/v1/users/{user_id} 获取用户信息

### POST /api/v1/tickets 创建工单

## 错误码
| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未认证 |
| 403 | 无权限 |
| 404 | 资源不存在 |
| 429 | 请求频率超限 |
| 500 | 服务器内部错误 |

## AUTH_401 错误处理
当 SDK 接入遇到 AUTH_401 错误时：
1. 检查 API Key 是否已过期
2. 确认 API Key 在请求头中正确传递
3. 验证该 Key 是否有访问目标资源的权限
4. 如 Key 已过期，登录管理控制台重新生成
5. 联系管理员检查 IP 白名单配置
